import argparse
import faiss
import json
import os
from sklearn.preprocessing import normalize
from tqdm import tqdm
import numpy as np
from pseudo_decomp_utils import load_nlp, DATA_DIR, get_squad_path, get_hotpot_path, get_simple_mined_questions_path


def vectorize(nlp, question, strip_qmarks, lowercase):
    inp_question = question
    question = question.strip().replace(' ?', '?')
    if '?' not in question:
        question += '?'

    if strip_qmarks:
        question = question.strip('?').strip()
    if lowercase:
        question = question.lower()

    token_question = nlp(question)
    token_question2 = nlp(inp_question.strip().replace(' ?', '?'))
    v = token_question.vector * max(len(token_question), 1)
    return inp_question, question, v, len(token_question2)


def load_compositional_raw_qs(split):
    with open(get_hotpot_path(split)) as f:
        data_hard = json.load(f)
    raw_qs_hard = []

    for qa in data_hard:
        raw_q_hard = qa['question'].strip()
        raw_qs_hard.append(raw_q_hard)
    return raw_qs_hard


def load_compositional_questions(nlp, split, strip_qmarks=True, lowercase=False):
    print('Loading hard questions...')
    with open(get_hotpot_path(split)) as f:
        data_hard = json.load(f)
    qs_hard = []
    raw_qs_hard = []

    for qa in tqdm(data_hard):
        raw_q_hard = qa['question'].strip()
        raw_q_hard, q, v, qlen = vectorize(nlp, raw_q_hard, strip_qmarks=strip_qmarks, lowercase=lowercase)
        raw_qs_hard.append(raw_q_hard)
        qs_hard.append(v)
    qs_hard = np.array(qs_hard)
    print(f'Loaded {len(qs_hard)} hard questions!')

    return raw_qs_hard, qs_hard


def load_single_hop_qs(nlp, n_mined, min_q_len, max_q_len, strip_qmarks=True, lower_case=False, use_mined_questions=False):

    cache_path = os.path.join(DATA_DIR, f'cached_fasttext_simples.n_mined={n_mined}.min_q_len={min_q_len}.max_q_len={max_q_len}.strip_q_marks={strip_qmarks}.lower_case={lower_case}')
    if os.path.exists(cache_path + '.npy'):
        with open(cache_path + '.json') as f:
            raw_qs = json.load(f)
        qs = np.load(cache_path + '.npy')
        return raw_qs, qs

    print('Loading easy questions...')
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)

    qs = []
    raw_qs = []
    for article in tqdm(data_easy['data']):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question'].strip()
                raw_q, q, v, qlen = vectorize(nlp, raw_q, strip_qmarks=strip_qmarks, lowercase=lower_case)
                if min_q_len <= qlen <= max_q_len:
                    raw_qs.append(raw_q)
                    qs.append(v)

    if use_mined_questions:
        for n, line in tqdm(enumerate(open(get_simple_mined_questions_path()))):
            if n > n_mined:
                break
            raw_q = line.strip()
            raw_q, q, v, qlen = vectorize(nlp, raw_q, strip_qmarks=strip_qmarks, lowercase=lower_case)
            if min_q_len <= qlen <= max_q_len:
                raw_qs.append(raw_q)
                qs.append(v)

    qs = np.array(qs)
    print(f'Loaded {len(qs)} easy questions!')

    with open(cache_path + '.json', 'w') as f:
        json.dump(raw_qs, f)
    np.save(file=cache_path + '.npy', arr=qs)

    return raw_qs, qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev'], type=str,
                        help="Find NNs for which HotpotQA split?")
    parser.add_argument("--min_q_len", default=4, type=int,
                        help="Minimum number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--beam_size", default=100, type=int,
                        help="Top-K most similar questions to comp Q to consider")
    parser.add_argument("--max_q_len", default=20, type=int,
                        help="Max number of spacy tokens allowed in a SQuAD question")
    parser.add_argument("--n_mined", default=0, type=int,
                        help="Number of mined questions to use")
    parser.add_argument("--target_question_length", type=int, default=2)
    parser.add_argument("--data_folder", type=str, help="Path to save results to")
    args = parser.parse_args()
    args.lower_case = False
    args.use_mined_questions = False

    nlp = load_nlp()

    raw_hard_qs = load_compositional_raw_qs(args.split)

    # Load questions and convert to L2-normalized vectors
    raw_qs, qs = load_single_hop_qs(nlp, args.n_mined, args.min_q_len,
                                    args.max_q_len, strip_qmarks=False,
                                    lower_case=args.lower_case, use_mined_questions=args.use_mined_questions)
    raw_qs_stripped, qs_stripped = load_single_hop_qs(nlp, args.n_mined, args.min_q_len, args.max_q_len,
                                                      strip_qmarks=True, lower_case=args.lower_case)

    normed_qs = normalize(qs, axis=1, norm='l2')
    candidate_index = faiss.IndexFlatIP(300)
    candidate_index.add(normed_qs)

    solutions = []
    solution_scores = []

    def nn_search(query, search_space, cached_square_norms=None):
        if cached_square_norms is None:
            cached_square_norms = np.sum(search_space, axis=1)
        return np.sqrt(np.sum(query ** 2) + cached_square_norms - 2 * np.dot(search_space, query[0]))

    all_cached_square_norms = np.sum(qs_stripped**2, axis=1)
    k = args.beam_size
    faiss_batch_size = 500

    for bn in tqdm(range(0, len(raw_hard_qs), faiss_batch_size)):
        raw_hard_qs_batch = raw_hard_qs[bn: bn + faiss_batch_size]
        fbs = len(raw_hard_qs_batch)

        hard_q_stripped_batch = np.zeros((fbs, 300), dtype=np.float32)
        normed_hard_q_batch = np.zeros((fbs, 300), dtype=np.float32)

        for i, raw_hard_q in enumerate(raw_hard_qs_batch):
            _, _, hard_q, _ = vectorize(nlp, raw_hard_q, strip_qmarks=False, lowercase=args.lower_case)
            _, _, hard_q_stripped, _ = vectorize(nlp, raw_hard_q, strip_qmarks=True, lowercase=args.lower_case)
            normed_hard_q = normalize(hard_q[np.newaxis, :], axis=1, norm='l2')
            hard_q_stripped_batch[i] = hard_q_stripped
            normed_hard_q_batch[i] = normed_hard_q

        Dts, Its = candidate_index.search(normed_hard_q_batch, k)

        for i, raw_hard_q in enumerate(raw_hard_qs_batch):
            hard_q_stripped = hard_q_stripped_batch[i: i+1]
            Dt, It = Dts[i: i+1], Its[i: i+1]

            qs_stripped_top_k = qs_stripped[It[0]]
            cached_square_norms = all_cached_square_norms[It[0]]

            zero_hop_scores = nn_search(hard_q_stripped, qs_stripped_top_k, cached_square_norms=cached_square_norms)

            solution_list = np.zeros((k, args.target_question_length + 1), dtype=np.int32)  # (N, )
            solution_list[:, 0] = It[0]
            solution_scores_list = np.zeros((k, args.target_question_length + 1))
            solution_scores_list[:, 0] = zero_hop_scores
            old_solution_vecs = qs_stripped_top_k.copy()

            n_solution_iters = 1
            while n_solution_iters < (args.target_question_length + 1):
                for solution_index in range(k):
                    old_solution_vec = old_solution_vecs[solution_index]
                    query = hard_q_stripped - old_solution_vec
                    total_scores = nn_search(query, qs_stripped_top_k, cached_square_norms=cached_square_norms)
                    total_scores[solution_index] = 1e6

                    best_temp = np.argmin(total_scores)
                    best = It[0, best_temp]
                    best_score = total_scores[best_temp]

                    solution_list[solution_index, n_solution_iters] = best
                    solution_scores_list[solution_index, n_solution_iters] = best_score
                    old_solution_vecs[solution_index] += qs_stripped_top_k[best_temp]

                n_solution_iters += 1

            # prevent ones
            solution_scores_list[:, 0] = 1e9

            best_i, best_j = np.unravel_index(solution_scores_list.argmin(), solution_scores_list.shape)
            best_sol, best_score = solution_list[best_i, :best_j + 1], solution_scores_list[best_i, best_j]
            solutions.append([raw_qs[i] for i in best_sol])
            solution_scores.append(best_score)

    print('Saving to file...')
    os.makedirs(args.data_dir, exist_ok=True)
    save_split = 'valid' if args.split == 'dev' else 'train'
    with open(f'{args.data_dir}/{save_split}.sh', 'w') as f:
        f.writelines('\n'.join([' '.join(sqs) for sqs in solutions]) + '\n')
    with open(f'{args.data_dir}/{save_split}.lens', 'w') as f:
        f.writelines('\n'.join([str(len(sqs)) for sqs in solutions]) + '\n')
    with open(f'{args.data_dir}/{save_split}.scores', 'w') as f:
        f.writelines('\n'.join([str(s) for s in solution_scores]) + '\n')
    with open(f'{args.data_dir}/{save_split}.mh', 'w') as f:
        f.writelines('\n'.join(raw_hard_qs) + '\n')
    print(f'Done! Saved to {args.data_dir}/{save_split}.sh mh')


if __name__ == '__main__':
    main()

