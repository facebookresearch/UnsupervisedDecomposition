# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from transformers import BertModel, BertTokenizer
import json
import numpy as np
import logging
from pseudo_decomp_utils import get_hotpot_path, get_squad_path, get_bert_embedding_path, get_simple_mined_questions_path

DEBUG = False
MODEL_CLASS = BertModel
TOKENIZER_CLASS = BertTokenizer
PRETRAINED_WEIGHTS = 'bert-large-cased'
N_LAYERS = 24
BATCH_SIZE = 16
INCLUDE_CLS_AND_SEP = True
USE_CLS = False


def embed_muppet(raw_qs, savename):
    tokenizer = TOKENIZER_CLASS.from_pretrained(PRETRAINED_WEIGHTS)
    model = MODEL_CLASS.from_pretrained(PRETRAINED_WEIGHTS,  output_hidden_states=True, torchscript=True)
    model.eval()
    model.cuda()

    vector_reps = {i: [] for i in range(N_LAYERS)}
    lens = []
    with torch.no_grad():
        for i in range(0, len(raw_qs), BATCH_SIZE):
            if i % 100 == 0:
                logging.info(f'processed {i}')
            q_batch = raw_qs[i: i + BATCH_SIZE]
            input_ids = [
                tokenizer.encode(q, add_special_tokens=True)
                for q in q_batch
            ]
            input_ids = [t[:512] for t in input_ids]
            lengths = [len(inp) for inp in input_ids]
            lens += lengths
            max_l = max(lengths)
            input_tensor = torch.tensor([
                inp + [tokenizer.pad_token_id for _ in range(max_l - len(inp))]
                for inp in input_ids]
            )
            attention_mask = (input_tensor == tokenizer.pad_token_id)
            attention_mask = 1 - attention_mask.to(torch.int)
            _, _, all_hidden_states = model(input_ids=input_tensor.cuda(), attention_mask=attention_mask.cuda())

            for layer in range(N_LAYERS):
                for sent in range(len(q_batch)):
                    if INCLUDE_CLS_AND_SEP:
                        vector = all_hidden_states[layer][sent, :lengths[sent]].sum(0)
                    elif USE_CLS:
                        vector = all_hidden_states[layer][sent, 0]
                    else:
                        vector = all_hidden_states[layer][sent, 1:lengths[sent]-2].sum(0)


                    vector_reps[layer].append(vector.cpu())

            if DEBUG:
                if i > 10000:
                    break

    for layer, reps in vector_reps.items():
        matrix = torch.stack(reps).numpy()
        np.save(savename + f'.{layer}', matrix)
    np.save(savename + '.lengths', lens)


def load_squad_qs():
    with open(get_squad_path('train')) as f:
        data_easy = json.load(f)

    raw_qs = []

    for article in data_easy['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                raw_q = qa['question'].strip()
                if '?' not in raw_q:
                    raw_q += '?'
                raw_qs.append(raw_q)
    return raw_qs


def load_hotpot_qs(split):
    hp_path = get_hotpot_path(split)
    with open(hp_path) as f:
        data_hard = json.load(f)
    raw_qs_hard = []
    for qa in data_hard:
        raw_q_hard = qa['question'].strip()
        raw_qs_hard.append(raw_q_hard)
    return raw_qs_hard


def load_simple_questions(input_file):
    qs = []
    for line in open(input_file):
        qs.append(line.strip().replace(' ?', '?'))
    return qs


def embed_squad():
    qs = load_squad_qs()
    embed_muppet(qs, get_bert_embedding_path('squad', 'train'))


def embed_hotpot():
    qs = load_hotpot_qs('train')
    embed_muppet(qs,  get_bert_embedding_path('hotpotqa', 'train'))
    qs = load_hotpot_qs('valid')
    embed_muppet(qs,  get_bert_embedding_path('hotpotqa', 'valid'))


def embed_simple_questions():
    qs = load_simple_questions(get_simple_mined_questions_path())
    embed_muppet(qs,  get_bert_embedding_path('simple_mined_questions', 'train'))


if __name__ == '__main__':
    embed_squad()
    embed_hotpot()
