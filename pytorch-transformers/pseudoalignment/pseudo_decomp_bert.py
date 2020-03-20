# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import faiss
import json
from sklearn.preprocessing import normalize
from tqdm import tqdm
import numpy as np
from pseudo_decomp_utils import dump_pseudoalignments, decomp,\
    get_squad_path, get_hotpot_path, get_simple_mined_questions_path, get_bert_embedding_path
import os


def main_bert():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev'], type=str,
                        help="Find NNs for which HotpotQA split?")
    parser.add_argument("--min_q_len", default=4, type=int,
                        help="Minimum number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--beam_size", default=100, type=int,
                        help="Top-K most similar questions to comp Q to consider")
    parser.add_argument("--max_q_len", default=20, type=int,
                        help="Max number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--layer", default=0, type=int,
                        help="bert layer to use")
    parser.add_argument("--data_folder", type=str, help="Path to save results to")
    args = parser.parse_args()
    args.use_mined_qs = False

    # Load questions and convert to L2-normalized vectors
    print('Loading easy questions...')
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)

    if not os.path.exists(get_bert_embedding_path('squad', 'train', args.layer)):
        raise Exception(f"BERT embeddings not found at {get_bert_embedding_path('squad', 'train', args.layer)}"
                        ", run `embed_questions_with_bert.py`")

    qs_all = np.load(get_bert_embedding_path('squad', 'train', args.layer))
    qs_inds = []
    raw_qs = []
    c = 0
    for article in tqdm(data_easy['data']):
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

    if args.use_mined_qs:
        mined_qs_all = np.load(get_bert_embedding_path('simple_minded_questions', 'train', args.layer))
        mined_qs_inds = []
        for n, line in tqdm(enumerate(open(get_simple_mined_questions_path()))):
            if n > 10000000:
                break
            raw_q = line.strip().replace(' ?', '?')
            if args.min_q_len <= len(raw_q.split()) <= args.max_q_len:
                raw_qs.append(raw_q)
                mined_qs_inds.append(n)
        mined_qs = mined_qs_all[mined_qs_inds]
        qs = np.concatenate([qs, mined_qs], 0)

    print(f'Loaded {len(qs)} easy questions!')

    print('Loading hard questions...')
    qs_hard = np.load(get_bert_embedding_path('hotpotqa', args.split, args.layer))
    hq = get_hotpot_path(args.split)
    with open(hq) as f:
        data_hard = json.load(f)

    raw_qs_hard = []
    for qa in tqdm(data_hard):
        raw_q_hard = qa['question'].strip()
        raw_qs_hard.append(raw_q_hard)

    print(f'Loaded {len(qs_hard)} hard questions!')

    print('Indexing easy Qs...')
    index = faiss.IndexFlatIP(qs.shape[1])  # L2 Norm then Inner Product == Cosine Similarity
    normed_qs = normalize(qs, axis=1, norm='l2')
    normed_qs_hard = normalize(qs_hard, axis=1, norm='l2')
    index.add(normed_qs)
    print(f'Total Qs indexed: {index.ntotal}')

    print('SQ1 NN search...')
    beam_q_pairs, beam_sq1, beam_sq2 = decomp(raw_qs, raw_qs_hard, normed_qs_hard, normed_qs, index, args.beam_size)

    print('Saving to file...')
    save_split = 'valid' if args.split == 'dev' else 'train'
    dump_pseudoalignments(args.data_folder, save_split, raw_qs_hard, beam_sq1, beam_sq2, beam_q_pairs)


if __name__ == '__main__':
    main_bert()
