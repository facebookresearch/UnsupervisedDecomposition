import argparse
import json
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from pseudo_decomp_utils import dump_pseudoalignments, get_squad_path, get_hotpot_path


def main_tfidf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev'], type=str,
                        help="Find NNs for which HotpotQA split?")
    parser.add_argument("--min_q_len", default=4, type=int,
                        help="Minimum number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--beam_size", default=100, type=int,
                        help="Top-K most similar questions to comp Q to consider")
    parser.add_argument("--max_q_len", default=20, type=int,
                        help="Max number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--data_folder", type=str, help="Path to save results to")
    args = parser.parse_args()


    # Load questions and convert to L2-normalized vectors
    print('Loading easy questions...')
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)

    raw_qs = []
    for article in tqdm(data_easy['data']):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question'].strip()
                if '?' not in raw_q:
                    raw_q += '?'
                if args.min_q_len <= len(raw_q.split()) <= args.max_q_len:
                    raw_qs.append(raw_q)

    print(f'Loaded {len(raw_qs)} easy questions!')

    print('Loading hard questions...')
    with open(get_hotpot_path(args.split)) as f:
        data_hard = json.load(f)

    raw_qs_hard = []
    for qa in tqdm(data_hard):
        raw_q_hard = qa['question'].strip()
        raw_qs_hard.append(raw_q_hard)

    print(f'Loaded {len(raw_qs_hard)} hard questions!')

    print('Building Index:')

    all_qs_raw = raw_qs_hard + raw_qs

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                         min_df=0, sublinear_tf=True, norm=None)#"l2")
    tf.fit_transform(all_qs_raw)

    normed_qs = tf.transform(raw_qs)
    normed_qs_hard = tf.transform(raw_qs_hard)
    print('SQ1 NN search...')

    beam_q_pairs = []
    beam_sq1, beam_sq2 = [], []
    batch_size, beam_size = 100, args.beam_size
    n_not_top = 0
    for batch_start in tqdm(range(0, len(raw_qs_hard), batch_size)):
        to_search = normed_qs_hard[batch_start: batch_start + batch_size]
        b, k, d = to_search.shape[0], beam_size, normed_qs.shape[1]

        scores = linear_kernel(to_search, normed_qs)
        indices = np.argpartition(scores, -beam_size, axis=1)[:, -beam_size:]
        values = scores[np.repeat(np.arange(scores.shape[0]), beam_size), indices.ravel()].reshape(scores.shape[0],
                                                                                                   beam_size)
        si_temp = np.argsort(-values, axis=1)
        It = indices[np.repeat(np.arange(b), beam_size), si_temp.ravel()].reshape(b, beam_size)
        Dt = values[np.repeat(np.arange(b), beam_size), si_temp.ravel()].reshape(b, beam_size)

        for i in range(b):
            nqs = normed_qs[It[i]]
            self_sims = linear_kernel(nqs, nqs)
            temp_mat = np.repeat(Dt[i, :, np.newaxis], k, axis=1)
            hard_q_sims = temp_mat + temp_mat.T
            objective_scores = hard_q_sims - self_sims
            sq1_ind_temp, sq2_ind_temp = np.unravel_index(np.argmax(objective_scores), objective_scores.shape)
            sq1_ind, sq2_ind = It[i, sq1_ind_temp], It[i, sq2_ind_temp]
            if sq1_ind_temp != 0 and sq2_ind_temp != 0:
                n_not_top += 1

            beam_q_pairs.append(raw_qs[sq1_ind] + ' ' + raw_qs[sq2_ind])
            beam_sq1.append(raw_qs[sq1_ind])
            beam_sq2.append(raw_qs[sq2_ind])


    print('Saving to file...')
    save_split = 'valid' if args.split == 'dev' else 'train'
    dump_pseudoalignments(args.data_folder, save_split, raw_qs_hard, beam_sq1, beam_sq2, beam_q_pairs)


if __name__ == "__main__":
    main_tfidf()

