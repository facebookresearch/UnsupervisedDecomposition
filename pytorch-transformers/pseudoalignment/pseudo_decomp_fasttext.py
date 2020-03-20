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
from pseudo_decomp_utils import load_nlp, dump_pseudoalignments, decomp, get_squad_path,\
    get_simple_mined_questions_path, get_hotpot_path


def main_fasttext():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev'], type=str,
                        help="Find NNs for which HotpotQA split?")
    parser.add_argument("--min_q_len", default=4, type=int,
                        help="Minimum number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--beam_size", default=100, type=int,
                        help="Top-K most similar questions to comp Q to consider")
    parser.add_argument("--max_q_len", default=20, type=int,
                        help="Max number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--data_folder", default='all.pseudo_decomp', type=str, help="Path to save results to")
    args = parser.parse_args()
    args.use_mined_qs = False

    # Load word embeddings
    print('Loading word embeddings...')
    nlp = load_nlp()

    # Load questions and convert to L2-normalized vectors
    print('Loading easy questions...')
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)
    qs = []
    raw_qs = []
    for article in tqdm(data_easy['data']):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question'].strip()
                if '?' not in raw_q:
                    raw_q += '?'
                token_q = nlp(raw_q)
                if args.min_q_len <= len(token_q) <= args.max_q_len:
                    raw_qs.append(raw_q)
                    qs.append(token_q.vector * max(len(token_q), 1))

    if args.use_mined_qs:
        for n, line in tqdm(enumerate(open(get_simple_mined_questions_path()))):
            if n > 10000000:
                break
            raw_q = line.strip().replace(' ?', '?')
            token_q = nlp(raw_q)
            if args.min_q_len <= len(token_q) <= args.max_q_len:
                raw_qs.append(raw_q)
                qs.append(token_q.vector * max(len(token_q), 1))

    qs = np.array(qs)
    print(f'Loaded {len(qs)} easy questions!')

    print('Loading hard questions...')
    with open(get_hotpot_path(args.split)) as f:
        data_hard = json.load(f)

    qs_hard = []
    raw_qs_hard = []
    for qa in tqdm(data_hard):
        raw_q_hard = qa['question'].strip()
        token_q_hard = nlp(raw_q_hard)
        raw_qs_hard.append(raw_q_hard)
        qs_hard.append(token_q_hard.vector * max(len(token_q_hard), 1))
    qs_hard = np.array(qs_hard)
    print(f'Loaded {len(qs_hard)} hard questions!')

    print('Indexing easy Qs...')
    index = faiss.IndexFlatIP(300)  # L2 Norm then Inner Product == Cosine Similarity
    normed_qs = normalize(qs, axis=1, norm='l2')
    normed_qs_hard = normalize(qs_hard, axis=1, norm='l2')
    index.add(normed_qs)
    print(f'Total Qs indexed: {index.ntotal}')

    print('SQ1 NN search...')
    beam_q_pairs, beam_sq1, beam_sq2 = decomp(raw_qs, raw_qs_hard, normed_qs_hard, normed_qs, index, args.beam_size)

    save_split = 'valid' if args.split == 'dev' else 'train'
    print('Saving to file...')
    dump_pseudoalignments(args.data_folder, save_split, raw_qs_hard, beam_sq1, beam_sq2, beam_q_pairs)


if __name__ == '__main__':
    main_fasttext()
