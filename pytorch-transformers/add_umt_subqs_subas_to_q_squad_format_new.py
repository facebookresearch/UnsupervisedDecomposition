# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import numpy as np
import os
import random
from nltk import sent_tokenize
from tqdm import tqdm


DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'XLM/data')


def read_lines_from_path(path):
    """
    Utility to read stripped lines from specified filepath
    """
    with open(path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def find_good_qids(args):
    print('Finding good QIDs...')
    good_qids = set([])
    total_qids = 0
    good_inps, good_hyps = [], []
    for split in args.splits:
        if ('adversarial' in split) and ('dev' in split):
            split = 'dev'  # Add Good QIDs using the dev set
        split = split.replace('dev', 'valid')
        hyp_lines = read_lines_from_path(f'{args.model_dir}/hyp.st={args.sample_temperature}.bs={args.beam}.lp={args.length_penalty}.es=False.seed={args.seed}.mh-sh.{split}.pred.bleu.sh.txt')
        input_lines = read_lines_from_path(f'data/umt/all/{split}.mh.tok')
        qids_list = read_lines_from_path(f'data/umt/all/{split}.qids.txt')
        total_qids += len(qids_list)

        doubles, contains, unchanged, too_few_qs, too_many_qs, all_q_words_in_subq, subq_longer_than_q, bads = 0, 0, 0, 0, 0, 0, 0, 0
        for inp, hyp, qid in tqdm(zip(input_lines, hyp_lines, qids_list)):
            bad = False
            if hyp.count('?') == 2:
                l, r, _ = hyp.split('?')
                l = l + '?'
                r = r + '?'
                if l == r:
                    doubles += 1
                    bad = True  # Unnecessary to use doubles for the "bad" criteria
                l_toks = l.split()
                r_toks = r.split()
                inp_toks = inp.split()
                for subq_toks in [l_toks, r_toks]:
                    if set(inp_toks).issubset(set(subq_toks)):
                        all_q_words_in_subq += 1
                        bad = True
                        break
                for subq_toks in [l_toks, r_toks]:
                    if len(subq_toks) >= len(inp_toks):
                        subq_longer_than_q += 1
                        bad = True
                        break
            elif hyp.count('?') < 2:
                too_few_qs += 1
                bad = True
            else:
                too_many_qs += 1
                if not args.one_to_variable:
                    bad = True
            if inp in hyp:
                contains += 1
                bad = True
                if inp == hyp:
                    unchanged += 1
            bads += bad
            if not bad:
                good_inps.append(inp)
                good_hyps.append(hyp)
                good_qids.add(qid)

        print('doubles', doubles)
        print('contains', contains)
        print('unchanged', unchanged)
        print('too_few_qs', too_few_qs)
        print('too_many_qs', too_many_qs)
        print('all_q_words_in_subq', all_q_words_in_subq)
        print('subq_longer_than_q', subq_longer_than_q)
        print('bads', bads)
        print('% Good QIDs:', round(100. * len(good_qids) / total_qids, 2))
    return good_qids, good_hyps, good_inps


def get_answer_sentence_no(evidence_sents):
    for evidence_sent_no, evidence_sent in enumerate(evidence_sents):
        if '@@' in evidence_sent:
            return evidence_sent_no
    return 0


def get_answer_sentences_with_window(evidence_paragraph, num_sentences, centering):
    # Find answer sentence alone
    assert centering in ['center', 'left', 'right'], f'centering={centering} not supported'
    evidence_sents = sent_tokenize(evidence_paragraph)
    evidence_sent_no = get_answer_sentence_no(evidence_sents)
    # Find window around answer sentence
    if centering == 'right':
        start_sent_shift = 0
    elif centering == 'left':
        start_sent_shift = num_sentences - 1
    elif centering == 'center':
        start_sent_shift = num_sentences // 2  # 0 for left preference, +1 for right preference
    start_sent_no = max(0, evidence_sent_no - start_sent_shift)
    end_sent_no = min(len(evidence_sents), start_sent_no + num_sentences)
    start_sent_no = max(0, end_sent_no - num_sentences)
    assert len(evidence_sents[start_sent_no: end_sent_no]) <= min(len(evidence_sents), num_sentences), f'Expected <={min(len(evidence_sents), num_sentences)} sentences but got {len(evidence_sents[start_sent_no: end_sent_no])} (range: [{start_sent_no}, {end_sent_no})) for paragraph {evidence_paragraph}'
    # Post-process text
    if (' '.join(evidence_sents[start_sent_no: end_sent_no])).count('@@') == 1:
        end_sent_no += 1  # Include next sentence if span crosses sentence boundary
    answer_sents = ' '.join(evidence_sents[start_sent_no: end_sent_no])
    if answer_sents.count('@@') == 1:
        answer_sents += ' @@'
    return answer_sents


def get_answer_text(answer_data, args, random_seed):
    if args.atype == 'predicted':
        answer_text = answer_data['text']
    elif args.atype == 'random':  # Replace predicted answer with random entity from passage
        answer_text = random.Random(random_seed).choice(qid2passage_entities[qid])
    elif args.atype == 'paragraph':
        answer_text = answer_data['evidence'].split('</title>', 1)[-1]
        if not args.show_span:
            answer_text = answer_text.replace('@@ ', '').replace(' @@', '').replace('@@', '')
    elif args.atype.startswith('sentence'):
        answer_text = get_answer_sentences_with_window(answer_data['evidence'].split('</title>', 1)[-1], args.num_sentences, args.centering)
        if not args.show_span:
            answer_text = answer_text.replace('@@ ', '').replace(' @@', '').replace('@@', '')
    else:
        raise NotImplementedError(f'args.atype {args.atype}')
    # Handle yes/no sub-answers
    if args.subq_model != 'decomprc':
        if answer_data['text'].strip() == 'yes':
            answer_text = answer_text.replace('no ', '', 1)
        elif answer_data['text'].strip() == 'no':
            answer_text = answer_text.replace('yes ', '', 1)
        else:
            answer_text = answer_text.replace('yes no ', '', 1)
    return answer_text.strip()


parser = argparse.ArgumentParser()
parser.add_argument("--subqs_dir",  # e.g. data/hotpot-all.umt.all.model=18635857.st=0.0.beam=4.lp=0.6.seed=0
                    required=True,
                    type=str,
                    help="Directory containing UMT-Generated Sub-Qs.")
parser.add_argument("--splits",
                    required=True,
                    nargs='+',
                    help="Dataset splits to process (e.g., dev).")
parser.add_argument("--num_subas",
                    default=1,
                    type=int,
                    help="Number of Sub-answers per Sub-Q.")
parser.add_argument("--num_shards",
                    default=1,
                    type=int,
                    help="Number of shards for answers.")
parser.add_argument("--subsample_data",
                    action='store_true',
                    default=False,
                    help="Make subsets of train with fewer Sub-Qs?")
parser.add_argument("--use_squad",
                    action='store_true',
                    default=False,
                    help="Add SQuAD Qs to training set?")
parser.add_argument("--use_easy",
                    action='store_true',
                    default=False,
                    help="Add easy Hotpot Qs to training set?")
parser.add_argument("--use_q",
                    action='store_true',
                    default=False,
                    help="Add SubQs?")
parser.add_argument("--use_a",
                    action='store_true',
                    default=False,
                    help="Add predicted answer to original question?")
parser.add_argument("--use_subq",
                    action='store_true',
                    default=False,
                    help="Add SubQs?")
parser.add_argument("--use_suba",
                    action='store_true',
                    default=False,
                    help="Add Sub-Answers?")
parser.add_argument("--atype",
                    required=True,
                    type=str,
                    help="What kind of sub-answer to add (if any), i.e. 'predicted', 'random', 'sentence', 'paragraph'")
parser.add_argument("--show_span",
                    action='store_true',
                    default=False,
                    help="Show span of predicted answer/sub-answer?")
parser.add_argument("--no_bad_subqs",
                    action='store_true',
                    default=False,
                    help="Skip adding bad sub-qs to HotpotQA input?")
parser.add_argument("--model_dir",  # Required with --no_bad_subqs
                    default=None,
                    type=str,
                    help="UMT model directory.")
parser.add_argument("--sample_temperature",  # Required with --no_bad_subqs
                    default=0.0,
                    type=float,
                    help="UMT generations sampling temperature.")
parser.add_argument("--beam",  # Required with --no_bad_subqs
                    default=5,
                    type=int,
                    help="UMT generations beam size.")
parser.add_argument("--length_penalty",  # Required with --no_bad_subqs
                    default=1.0,
                    type=float,
                    help="UMT generations length penalty.")
parser.add_argument("--seed",  # Required with --no_bad_subqs
                    default=0,
                    type=int,
                    help="UMT generations seed.")
parser.add_argument("--one_to_variable",
                    action='store_true',
                    default=False,
                    help="Assume 1->2 mapping or 1->(variable #) mapping?")
parser.add_argument("--version_no",
                    default=0,
                    type=int,
                    help="Version number to append (doesn't append if 0)")
parser.add_argument("--subq_model",
                    default='decomprc',
                    type=str,  # E.g., 'decomprc', 'roberta-large', 'roberta-large-np=1-3-5'
                    help="What sub-question model to use.")
args = parser.parse_args()
assert not (args.no_bad_subqs and (args.model_dir is None)), '--model_dir is required to use --no-bad-subqs'

# Make the flag string for saved files
save_dir_parts = [args.subqs_dir]
if args.atype:
    save_dir_parts.append(f'atype={args.atype}')
if args.no_bad_subqs:
    save_dir_parts.append('no_bad_subqs')
if args.one_to_variable:
    save_dir_parts.append('one_to_variable')
if args.show_span:
    save_dir_parts.append('show_span')
if args.subq_model != 'decomprc':
    save_dir_parts.append(f'subq_model={args.subq_model}')
if args.use_a:
    save_dir_parts.append('use_a')
if args.use_q:
    save_dir_parts.append('use_q')
if args.use_suba:
    save_dir_parts.append('use_suba')
if args.use_subq:
    save_dir_parts.append('use_subq')
if args.version_no != 0:
    save_dir_parts.append(f'version_no={args.version_no}')
flags_string = ".".join(save_dir_parts)

# Load SQUAD formatted hotpot data
data = {}
for split in args.splits:
    with open(f'{DATA_DIR}/hotpot-squad/{split}.json') as f:
        data[split] = json.load(f)


if args.no_bad_subqs:
    good_qids, good_hyps, good_inps = find_good_qids(args)


# Read SubQs
subqs = {}
total_raw_subqs = 0
for split in args.splits:
    with open(f'{args.subqs_dir}/{split}.json') as f:
        raw_subqs = json.load(f)
    for example in raw_subqs['data']:
        for paragraph in example['paragraphs']:
            qid = paragraph['qas'][0]['id'].split('-')[0]
            subqs[qid] = [qa['question'] for qa in paragraph['qas']]
            total_raw_subqs += len(subqs[qid])


# Read SubAs
raw_subas = {}
for split in args.splits:
    sharded = not ((args.num_shards == 1) or (split == 'dev'))
    shard_str = f'num_shards={args.num_shards}.shard_no=$SHARD_NO.' if sharded else ''
    if args.subq_model == 'decomprc':
        suba_file = f'{args.subqs_dir}/bert_predict.nbest=10/{split}.{shard_str}nbest_predictions.json'
    elif args.subq_model.startswith('roberta-large'):  # E.g., roberta-large-np=5 or roberta-large-np=1-3-5
        subq_model_type = 'rs=0' if args.subq_model == 'roberta-large' else args.subq_model.split('roberta-large-')[1]
        suba_file = f'{args.subqs_dir}/roberta_predict.{subq_model_type}/hotpot_predictions_gn_info_{split}.{shard_str}json'
    if not sharded:
        with open(suba_file) as f:
            raw_subas.update(json.load(f))
    else:
        for shard_no in range(args.num_shards):
            with open(suba_file.replace('$SHARD_NO', shard_no)) as f:
                raw_subas.update(json.load(f))
max_subqs = 10 if args.one_to_variable else 2
subas = {}
total_subas = 0
qids = {k.split('-')[0] for k in raw_subas.keys()}
for qid in qids:
    if args.subq_model == 'decomprc':
        subas[qid] = [raw_subas[f'{qid}-{i}'] for i in range(max_subqs) if f'{qid}-{i}' in raw_subas]
    else:  # Add sub-answer to list (to adjust for different file structure)
        subas[qid] = [[raw_subas[f'{qid}-{i}']] for i in range(max_subqs) if f'{qid}-{i}' in raw_subas]
    total_subas += len(subas[qid])
print(f'total_raw_subqs={total_raw_subqs}, total_subas={total_subas}, mean subas/q={total_subas / len(qids)}')

# Read in original answers
if args.use_a:
    if args.subq_model == 'decomprc':
        origas = {}
        for split in args.splits:
            with open(f'{os.getenv("HOME")}/research/DecompRC/DecompRC/out/subqs_o/{split}_nbest_predictions.json') as f:
                raw_origas = json.load(f)
            for qid in raw_origas.keys():
                origas[qid] = raw_origas[qid]
    else:
        raise NotImplementedError(f'--use_a for --subq_model {args.subq_model}')

# For random entity answer ablation
with open(f'{DATA_DIR}/hotpot-orig/qid2passage_entities.json') as f:
    qid2passage_entities = json.load(f)

a_rank = 0
si = 0  # Or: for si in range(args.num_subas)
# sj = 0  # Or: for sj in range(args.num_subas)
suba_rank = [si] * max_subqs  # Or: [si, sj]

q_start = '//'
a_start = '/'

# Set variables for get_answer_sentences_with_window
atype_parts = args.atype.split('-')
args.num_sentences = int(atype_parts[1]) if len(atype_parts) > 1 else 1
args.centering = atype_parts[2] if len(atype_parts) > 2 else 'center'


question_augmenteds = []
for split in data.keys():
    for article_no in tqdm(range(len(data[split]['data']))):
        for paragraph_no in range(len(data[split]['data'][article_no]['paragraphs'])):
            if '_id' not in data[split]['data'][article_no]['paragraphs'][paragraph_no]:
                continue  # SQuAD question: skip
            qid = data[split]['data'][article_no]['paragraphs'][paragraph_no]['_id']
            for qa_no in range(len(data[split]['data'][article_no]['paragraphs'][paragraph_no]['qas'])):
                qa = data[split]['data'][article_no]['paragraphs'][paragraph_no]['qas'][qa_no]  # Pointer into data
                question_augmented = qa['question'].strip() if args.use_q else ''
                if args.use_a:
                    a_text = get_answer_text(origas[qid][a_rank], args, 41)
                    question_augmented += ' ' + a_start + ' ' + a_text
                    qa['probability_a'] = origas[qid][a_rank]['probability']
                if (not (args.no_bad_subqs and (qid not in good_qids))) and (qid in subqs):  # Don't add bad sub-qs
                    for subq_no, (subq, suba) in enumerate(zip(subqs[qid], subas[qid])):
                        if args.use_subq:
                            question_augmented += ' ' + q_start + ' ' + subq.strip()
                        if args.use_suba and (len(suba) > suba_rank[subq_no]):
                            suba_text = get_answer_text(suba[suba_rank[subq_no]], args, 42 + subq_no)
                            question_augmented += ' ' + a_start + ' ' + suba_text
                            qa['probability_subas'] = qa.get('probability_subas', [])
                            qa['probability_subas'].append(suba[suba_rank[subq_no]]['probability'])
                question_augmented = question_augmented.strip()
                qa['question'] = question_augmented
                question_augmenteds.append(question_augmented)

print('Example Augmented Qs:')
for question_augmented in question_augmenteds[::5][:6] + question_augmenteds[::5][-6:]:
    print(question_augmented + '\n')

qlens = np.array([len(q.split()) for q in question_augmenteds])
qlens.sort()
print('HotpotQA Mean # Q Tokens: ~', int(round(1.2 * qlens.mean())))
print('HotpotQA # Examples:', len(qlens))
print('HotpotQA 90 %-ile # Q Tokens:', int(round(1.2 * qlens[int(.9 * qlens.size)])))
print('HotpotQA 99 %-ile # Q Tokens:', int(round(1.2 * qlens[int(.99 * qlens.size)])))

save_dir = f'{flags_string}.suba1={suba_rank[0]}.suba2={suba_rank[1]}'
print(f'Saving to {save_dir}')
os.makedirs(save_dir, exist_ok=True)
for split in args.splits:
    with open(f'{save_dir}/{split}.json', 'w') as f:
        json.dump(data[split], f, indent=2)


# Generate (subset of) Q's
if not args.subsample_data:
    print('Done! (Not making subsamples of data)')
    exit()

max_exp = 9
data_basename = save_dir.split('/', 1)[1]
if args.use_squad:
    data_basename += '-squad'
if not args.use_easy:
    data_basename += '-no-easy'
for split in args.splits:
    print('Processing', data_basename, split)
    if split != 'train':  # Copy data for evaluation sets
        for exp in range(max_exp + 1):
            label_frac = (0.5 ** exp) if exp < max_exp else 0.
            frac_save_dir = f'{DATA_DIR}/{data_basename}.medium_hard_frac={label_frac}'
            os.makedirs(frac_save_dir, exist_ok=True)
            print(f'Saving to: {frac_save_dir}/{split}.json')
            with open(f'{frac_save_dir}/{split}.json', 'w') as f:
                json.dump(data[split], f, indent=2)
    else:  # Downsample medium/hard examples for training
        data_group = {'easy': [], 'medium': [], 'hard': [], 'squad': []}
        for article in data[split]['data']:
            data_group[article['paragraphs'][0].get('level', 'squad')].append(article)
        for group, articles in data_group.items():
            print(split, group, len(articles))

        random.Random(42).shuffle(data_group['medium'])
        random.Random(42).shuffle(data_group['hard'])

        for exp in range(max_exp + 1):
            label_frac = (0.5 ** exp) if exp < max_exp else 0.
            frac_save_dir = f'{DATA_DIR}/{data_basename}.medium_hard_frac={label_frac}'
            os.makedirs(frac_save_dir, exist_ok=True)
            new_data = {
                'data': (data_group['easy'] if args.use_easy else []) +
                         data_group['medium'][:round(len(data_group['medium']) * label_frac)] +
                         data_group['hard'][:round(len(data_group['hard']) * label_frac)] +
                        (data_group['squad'] if args.use_squad else []),
                'version': data[split]['version']
            }
            print('new_data samples', len(new_data['data']))
            print(f'Saving to: {frac_save_dir}/{split}.json')
            with open(f'{frac_save_dir}/{split}.json', 'w') as f:
                json.dump(new_data, f, indent=2)
