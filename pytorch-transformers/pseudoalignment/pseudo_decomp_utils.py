# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from spacy.language import Language
from tqdm import tqdm
import os
import numpy as np


DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'XLM/data')
SIMPLE_QUESTIONS_FILE = os.path.join(DATA_DIR, 'simple_mined_questions.txt')
PATH_TO_EMBEDDINGS_DUMP = os.path.join(DATA_DIR, 'pseudoalignments/embeddings/')
HOTPOT_TRAIN_PATH = os.path.join(DATA_DIR, 'hotpot-orig/hotpot_train_v1.json')
HOTPOT_DEV_PATH = os.path.join(DATA_DIR, 'hotpot-orig/hotpot_dev_distractor_v1.json')
SQUAD_TRAIN_PATH = os.path.join(DATA_DIR, 'squad/train-v2.0.json')
SQUAD_DEV_PATH = os.path.join(DATA_DIR, 'squad/dev-v2.0.json')
FASTTEXT_VECTORS_PATH = os.path.join(DATA_DIR, 'fastText/crawl-300d-2M.vec')


def get_squad_path(split):
    return SQUAD_TRAIN_PATH if split == 'train' else SQUAD_DEV_PATH


def get_hotpot_path(split):
    return HOTPOT_TRAIN_PATH if split == 'train' else HOTPOT_DEV_PATH


def get_simple_mined_questions_path():
    return f'{DATA_DIR}/simple_mined_questions.txt'


def get_bert_embedding_path(dataset, split, layer=None):
    if dataset == 'simple_mined_questions':
        stub = f'{PATH_TO_EMBEDDINGS_DUMP}/simple_mined_questions.bert_large'
    elif dataset == 'hotpotqa':
        stub = f'{PATH_TO_EMBEDDINGS_DUMP}/hotpot.{"train" if split == "train" else "valid"}.bert_large'
    elif dataset == 'squad':
        stub = f'{PATH_TO_EMBEDDINGS_DUMP}/squad.train.bert_large'
    else:
        raise Exception(f'Unrecognised dataset argument {dataset} ')
    if layer is not None:
        stub += f'.{layer}.npy'
    return stub


def get_fasttext_vectors_path():
    return FASTTEXT_VECTORS_PATH


def load_nlp():
    nlp = Language()

    with open(FASTTEXT_VECTORS_PATH, 'rb') as f:
        header = f.readline()
        nr_row, nr_dim = header.split()
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in tqdm(f, total=2000000):
            line = line.rstrip().decode("utf8")
            pieces = line.rsplit(" ", int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype="f")
            nlp.vocab.set_vector(word, vector)

    return nlp


def dump_pseudoalignments(save_dir, save_split, raw_qs_hard, sq1s, sq2s, sq_pairs):
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{save_split}.sh', 'w') as f:
        f.writelines('\n'.join(sq_pairs) + '\n')
    with open(f'{save_dir}/{save_split}.sh1', 'w') as f:
        f.writelines('\n'.join(sq1s) + '\n')
    with open(f'{save_dir}/{save_split}.sh2', 'w') as f:
        f.writelines('\n'.join(sq2s) + '\n')
    with open(f'{save_dir}/{save_split}.mh', 'w') as f:
        f.writelines('\n'.join(raw_qs_hard) + '\n')
    print(f'Done! Saved to {save_dir}/{save_split}.sh sh1 sh2 mh')


def decomp(raw_qs, raw_qs_hard, normed_qs_hard, normed_qs, index, beam_size):
    beam_q_pairs = []
    beam_sq1, beam_sq2 = [], []
    batch_size = 100

    for batch_start in tqdm(range(0, len(raw_qs_hard), batch_size)):
        to_search = normed_qs_hard[batch_start:batch_start + batch_size]
        b, k, d = len(to_search), beam_size, 300
        Dt, It = index.search(normed_qs_hard[batch_start:batch_start + batch_size], k)

        normed_qs_beam = normed_qs[It.flatten()].reshape((b, k, d))
        self_sims = (normed_qs_beam @ normed_qs_beam.transpose((0, 2, 1)))  # SQ1 dot SQ2

        temp_mat = np.repeat(Dt[:, :, np.newaxis], k, axis=2)
        hard_q_sims = temp_mat + temp_mat.transpose(0, 2, 1)  # Q dot SQ
        objective_scores = hard_q_sims - self_sims

        # find best:
        sq1_inds_temp, sq2_inds_temp = np.unravel_index([np.argmax(objective_scores[i]) for i in range(b)],
                                                        objective_scores[0].shape)
        # scores = obj[range(b), sq1_inds_temp, sq2_inds_temp]
        sq1_inds, sq2_inds = It[range(b), sq1_inds_temp], It[range(b), sq2_inds_temp]
        for sq1_ind, sq2_ind in zip(sq1_inds, sq2_inds):
            beam_q_pairs.append(raw_qs[sq1_ind] + ' ' + raw_qs[sq2_ind])
            beam_sq1.append(raw_qs[sq1_ind])
            beam_sq2.append(raw_qs[sq2_ind])

    return beam_q_pairs, beam_sq1, beam_sq2
