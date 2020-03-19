# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import apex
import os
import io
import sys
import argparse
import torch

from src.evaluation.evaluator import convert_to_text, eval_moses_bleu
from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp, restore_segmentation
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--ref_path", type=str, default="", help="Reference file path (if evaluating BLEU)")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--seed", type=str, default=0, help="Random seed (for sampled generations)")
    parser.add_argument("--amp", type=int, default=0, help="fp16 opt level (0 for fp32)")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # beam search
    parser.add_argument("--sample_temperature", type=float, default=0.0,
                        help="Beam size, default = None (greedy decoding)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    torch.manual_seed(params.seed)  # Set random seed. NB: Multi-GPU also needs torch.cuda.manual_seed_all(params.seed)
    assert (params.sample_temperature == 0) or (params.beam_size == 1), 'Cannot sample with beam search.'
    assert params.amp <= 1, f'params.amp == {params.amp} not yet supported.'
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=False).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    if all([k.startswith('module.') for k in reloaded['encoder'].keys()]):
        reloaded['encoder'] = {k[len('module.'):]: v for k, v in reloaded['encoder'].items()}
    encoder.load_state_dict(reloaded['encoder'])
    if all([k.startswith('module.') for k in reloaded['decoder'].keys()]):
        reloaded['decoder'] = {k[len('module.'):]: v for k, v in reloaded['decoder'].items()}
    decoder.load_state_dict(reloaded['decoder'])

    if params.amp != 0:
        models = apex.amp.initialize(
            [encoder, decoder],
            opt_level=('O%i' % params.amp)
        )
        encoder, decoder = models

    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    # read sentences from stdin
    src_sent = []
    for line in sys.stdin.readlines():
        assert len(line.strip().split()) > 0
        src_sent.append(line)
    logger.info("Read %i sentences from stdin. Translating ..." % len(src_sent))

    # f = io.open(params.output_path, 'w', encoding='utf-8')

    hypothesis = [[] for _ in range(params.beam_size)]
    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)

        # encode source batch and translate it
        encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)
        max_len = int(1.5 * lengths.max().item() + 10)
        if params.beam_size == 1:
            decoded, dec_lengths = decoder.generate(
                encoded, lengths.cuda(), params.tgt_id, max_len=max_len,
                sample_temperature=(None if params.sample_temperature == 0 else params.sample_temperature))
        else:
            decoded, dec_lengths, all_hyp_strs = decoder.generate_beam(
                encoded, lengths.cuda(), params.tgt_id, beam_size=params.beam_size,
                length_penalty=params.length_penalty,
                early_stopping=params.early_stopping,
                max_len=max_len,
                output_all_hyps=True
            )
        # hypothesis.extend(convert_to_text(decoded, dec_lengths, dico, params))

        # convert sentences to words
        for j in range(decoded.size(1)):

            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = src_sent[i + j].strip().replace('<unk>', '<<unk>>')
            target = " ".join([dico[sent[k].item()] for k in range(len(sent))]).replace('<unk>', '<<unk>>')
            if params.beam_size == 1:
                hypothesis[0].append(target)
            else:
                for hyp_rank in range(params.beam_size):
                    print(all_hyp_strs[j][hyp_rank if hyp_rank < len(all_hyp_strs[j]) else -1])
                    hypothesis[hyp_rank].append(all_hyp_strs[j][hyp_rank if hyp_rank < len(all_hyp_strs[j]) else -1])

            sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source.replace('@@ ', ''), target.replace('@@ ', '')))
            # f.write(target + "\n")

    # f.close()

    # export sentences to reference and hypothesis files / restore BPE segmentation
    save_dir, split = params.output_path.rsplit('/', 1)
    for hyp_rank in range(len(hypothesis)):
        hyp_name = f'hyp.st={params.sample_temperature}.bs={params.beam_size}.lp={params.length_penalty}.es={params.early_stopping}.seed={params.seed if (len(hypothesis) == 1) else str(hyp_rank)}.{params.src_lang}-{params.tgt_lang}.{split}.txt'
        hyp_path = os.path.join(save_dir, hyp_name)
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hypothesis[hyp_rank]) + '\n')
        restore_segmentation(hyp_path)

        # evaluate BLEU score
        if params.ref_path:
            bleu = eval_moses_bleu(params.ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, params.ref_path, bleu))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
