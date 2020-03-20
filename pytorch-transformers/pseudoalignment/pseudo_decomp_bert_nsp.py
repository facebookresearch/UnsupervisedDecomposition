# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import faiss
import json
from transformers import BertForNextSentencePrediction, BertTokenizer
import torch
from sklearn.preprocessing import normalize
import numpy as np
import logging
from pseudo_decomp_utils import dump_pseudoalignments, get_bert_embedding_path, get_squad_path, get_hotpot_path


DEBUG = False
MODEL_CLASS = BertForNextSentencePrediction
TOKENIZER_CLASS = BertTokenizer
PRETRAINED_WEIGHTS = 'bert-large-cased'
N_LAYERS = 24
BATCH_SIZE = 16


def main_bert_nsp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev'], type=str,
                        help="Find NNs for which HotpotQA split?")
    parser.add_argument("--min_q_len", default=4, type=int,
                        help="Minimum number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--max_q_len", default=20, type=int,
                        help="Max number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--layer", default=0, type=int,
                        help="layer of bert")
    parser.add_argument("--top_k_to_search", default=100, type=int)
    parser.add_argument("--top_k_to_pairwise_rank", default=20, type=int)
    parser.add_argument("--batch_num",  type=int, help="which batch to compute")
    parser.add_argument("--data_folder", type=str, help="Path to save results to")
    args = parser.parse_args()

    # Load questions and convert to L2-normalized vectors
    print('Loading easy questions...')
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)

    qs_all = np.load(get_bert_embedding_path('squad', 'train', args.layer))
    qs_inds = []
    raw_qs = []
    c = 0
    for article in data_easy['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question'].strip()
                if '?' not in raw_q:
                    raw_q += '?'
                if args.min_q_len <= len(raw_q.split()) <= args.max_q_len:
                    raw_qs.append(raw_q)
                    qs_inds.append(c)
                c += 1
    qs = qs_all[qs_inds]

    print('Loading hard questions...')
    qs_hard = np.load(get_bert_embedding_path('hotpotqa', args.split, args.layer))

    with open(get_hotpot_path(args.split)) as f:
        data_hard = json.load(f)

    raw_qs_hard = []
    for qa in data_hard:
        raw_q_hard = qa['question'].strip()
        raw_qs_hard.append(raw_q_hard)

    print(f'Loaded {len(qs_hard)} hard questions!')
    print('Indexing easy Qs...')
    index = faiss.IndexFlatIP(qs.shape[1])  # L2 Norm then Inner Product == Cosine Similarity
    normed_qs = normalize(qs, axis=1, norm='l2')
    normed_qs_hard = normalize(qs_hard, axis=1, norm='l2')
    index.add(normed_qs)
    print(f'Total Qs indexed: {index.ntotal}')

    tokenizer = TOKENIZER_CLASS.from_pretrained(PRETRAINED_WEIGHTS)
    model = MODEL_CLASS.from_pretrained(PRETRAINED_WEIGHTS, torchscript=True)
    model.eval()
    model.cuda()

    if DEBUG:
        raw_qs_hard = raw_qs_hard[:50]
    else:
        raw_qs_hard = raw_qs_hard[args.batch_num * 1000: args.batch_num * 1000 + 1000]

    mh_tokens_cache = {}

    for mh in raw_qs_hard:
        tokenized_mh = tokenizer.encode(mh, add_special_tokens=True)
        mh_tokens_cache[mh] = tokenized_mh

    k = args.top_k_to_search
    k2 = args.top_k_to_pairwise_rank

    def build_batch(long_q, q_batch):
        tokenized_mh = mh_tokens_cache[long_q]
        input_ids = [
            tokenized_mh + tokenizer.encode(q, add_special_tokens=True)[1:]
            for q in q_batch
        ]
        input_ids = [t[:512] for t in input_ids]
        max_l = max([len(inp) for inp in input_ids])
        input_tensor = torch.tensor([
            inp + [tokenizer.pad_token_id for _ in range(max_l - len(inp))]
            for inp in input_ids]
        )
        segment_tensor = torch.ones(input_tensor.shape, dtype=torch.int64)
        segment_tensor[:, :len(tokenized_mh)] = 0
        attention_mask = (input_tensor == tokenizer.pad_token_id)
        attention_mask = 1 - attention_mask.to(torch.int64)
        return input_tensor, segment_tensor, attention_mask

    nsp_sh1 = []
    nsp_sh2 = []
    nsp_pairs = []

    with torch.no_grad():
        for mh_ind, mh in enumerate(raw_qs_hard):
            if mh_ind % 10 == 0:
                logging.info(f'Completed {mh_ind} questions')
            single_score_map = {}
            Dt, It = index.search(normed_qs_hard[mh_ind:mh_ind + 1], k)
            raw_qs_to_search = [raw_qs[ind] for ind in It[0]]
            for b in range(0, len(raw_qs_to_search), BATCH_SIZE):
                short_q_batch = raw_qs_to_search[b:b + BATCH_SIZE]
                input_ids, segments, att_mask = build_batch(mh, short_q_batch)
                out = model(
                    input_ids=input_ids.cuda(),
                    token_type_ids=segments.cuda(),
                    attention_mask=att_mask.cuda()
                )
                batch_scores = torch.nn.functional.softmax(out[0], dim=1)[:, 0].cpu()
                for sq, c in zip(short_q_batch, list(batch_scores)):
                    single_score_map[sq] = c

            raw_q_to_search_2, _ = zip(*list(sorted(single_score_map.items(), key=lambda x: -x[1]))[:k2])

            pairs_to_search = []
            for sha in raw_q_to_search_2:
                for shb in raw_q_to_search_2:
                    if sha != shb:
                        pairs_to_search.append((sha, shb))

            pair_score_map = {}
            for b in range(0, len(pairs_to_search), BATCH_SIZE):
                pair_batch = pairs_to_search[b:b + BATCH_SIZE]
                input_ids, segments, att_mask = build_batch(mh, [a + ' ' + b for a, b in pair_batch])
                out = model(
                    input_ids=input_ids.cuda(),
                    token_type_ids=segments.cuda(),
                    attention_mask=att_mask.cuda()
                )
                batch_scores = torch.nn.functional.softmax(out[0], dim=1)[:, 0].cpu()
                for pair, pair_score in zip(pair_batch, list(batch_scores)):
                    pair_score_map[pair] = pair_score

            final_scores = []
            for sha, shb in pairs_to_search:
                final_scores.append((sha, shb, single_score_map[sha] + single_score_map[shb] + pair_score_map[(sha, shb)]))

            sha, shb, score = max(final_scores, key=lambda x: x[-1])
            nsp_sh1.append(sha)
            nsp_sh2.append(shb)
            nsp_pairs.append(sha + ' ' + shb)

    print('Saving to file...')
    save_split = 'valid' if args.split == 'dev' else 'train'
    dump_pseudoalignments(args.data_folder, save_split, raw_qs_hard, nsp_sh1, nsp_sh2, nsp_pairs)


if __name__ == '__main__':
    main_bert_nsp()
