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
from copy import deepcopy

DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'XLM/data')


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    required=True,
                    type=str,
                    help="UMT model directory.")
parser.add_argument("--data_folder",
                    required=True,
                    type=str,
                    help="UMT data directory (e.g., comparison.paired).")
parser.add_argument("--sample_temperature",
                    required=True,
                    type=float,
                    help="UMT generations sampling temperature.")
parser.add_argument("--beam",
                    required=True,
                    type=int,
                    help="UMT generations beam size.")
parser.add_argument("--length_penalty",
                    required=True,
                    type=float,
                    help="UMT generations length penalty.")
parser.add_argument("--seed",
                    required=True,
                    type=int,
                    help="UMT generations seed.")
parser.add_argument("--split",
                    required=True,  # e.g., train_para, train, valid, test, hotpot_dev_distractor_adversarial_jiang_v1
                    type=str,
                    help="Dataset split.")
parser.add_argument('--new_data_format', action='store_true',
                    help="Use pytorch-transformers data format (instead of DecompRC format)?")
args = parser.parse_args()

orig_data_folder = 'hotpot' if args.new_data_format else 'hotpot-all'
model_no = int(args.model_dir.split('/')[-1])
subqs_split = args.split
if ('adversarial' in args.split) and ('dev' in args.split):
    subqs_split = 'valid'
subqs_filename = 'hyp.st={}.bs={}.lp={}.es=False.seed={}.mh-sh.{}.pred.bleu.sh.txt'.format(args.sample_temperature, args.beam, args.length_penalty, args.seed, subqs_split)

subqs_filepath = os.path.join(os.environ['MAIN_DIR'], 'XLM', args.model_dir, subqs_filename)
hotpot_split = args.split.replace('valid', 'dev').replace('_para', '')

with open(subqs_filepath) as f:
    raw_subqs = f.readlines()
print('Read {} Sub-Q pairs'.format(len(raw_subqs)))
print('model_no={}, st={}, beam={}, length_penalty={}, seed={}, split={}'.format(model_no, args.sample_temperature, args.beam, args.length_penalty, args.seed, args.split))


subqs = []
for raw_subq in raw_subqs:
    ex_subqs = raw_subq.strip('\n').strip().split(' ?')
    proc_ex_subqs = []
    for ex_subq in ex_subqs:
        proc_ex_subq = ex_subq.strip()
        if len(proc_ex_subq) > 0:
            proc_ex_subqs.append(proc_ex_subq + '?')
    subqs.append(proc_ex_subqs)

subq_lens = np.array([len(subq) for subq in subqs])
print('Mean # of Sub-Qs:', round(subq_lens.mean(), 3))
subqs[0]


with open('{}/umt/{}/{}.qids.txt'.format(DATA_DIR, args.data_folder, subqs_split)) as f:
    qids = f.readlines()
qids = [qid.strip('\n') for qid in qids]
qid2subqs = {qid: subq_pair for qid, subq_pair in zip(qids, subqs)}
print('Read {} QIDs'.format(len(qids)))


with open('{}/{}/{}.json'.format(DATA_DIR, orig_data_folder, hotpot_split)) as f:
    data_hotpot = json.load(f)

num_hotpot_examples = len(data_hotpot['data'])
print('Read {} HotpotQA examples'.format(num_hotpot_examples))
qid2examples = {}
for example in data_hotpot['data']:
    qid = example['paragraphs'][0]['_id'] if args.new_data_format else example['paragraphs'][0]['qas'][0]['id']
    qid2examples[qid] = example


num_no_subqs = 0
new_data_hotpot = {'data': []}
for q_no, (qid, subqpair) in enumerate(qid2subqs.items()):
    if len(subqpair) == 0:
        subqpair = [example['paragraphs'][0]['qas'][0]['question']]
        print('No Sub-Qs. Using Original Q:', subqpair)
        num_no_subqs += 1
    example = deepcopy(qid2examples[qid])
    if args.new_data_format:  # New format: For each example, there's N paragraphs (qas appear once per paragraph)
        for p in range(len(example['paragraphs'])):
            example['paragraphs'][p]['qas'] = [
                {'question': subq,
                 'answers': [],
                 'id': qid + '-' + str(s) + '.' + str(p),
                 'is_impossible': True}
                for s, subq in enumerate(subqpair)]
    else:  # Old format: For each example, there's just one "paragraph" with a list of N paragraphs (qas appear once only)
        example['paragraphs'][0]['qas'] = [
            {'question': subq,
             'answers': [[] for _ in range(len(example['paragraphs'][0]['qas'][0]['answers']))],
             'id': qid + '-' + str(s)}
            for s, subq in enumerate(subqpair)]
    new_data_hotpot['data'].append(example)
print('# of Qs incorrectly without Sub-Qs:', num_no_subqs)


save_dir = '{}/{}.umt.{}.model={}.st={}.beam={}.lp={}.seed={}'.format(DATA_DIR, orig_data_folder, args.data_folder, model_no, args.sample_temperature, args.beam, float(args.length_penalty), args.seed)
print('Saving to', save_dir)
os.makedirs(save_dir, exist_ok=True)
with open('{}/{}.json'.format(save_dir, hotpot_split), 'w') as f:
    json.dump(new_data_hotpot, f, indent=2)
