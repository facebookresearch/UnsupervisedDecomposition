# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  RobertaConfig,
                                  RobertaForQuestionAnswering,
                                  RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule
# import sys
# sys.path.append('examples')

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from hotpot_evaluate_v1 import eval as evaluate_on_hotpot
import math
import pickle

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer, tb_writer):
    """ Train the model """
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if global_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.save_freq is not None:
        args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps // args.save_freq

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta_1, args.adam_beta_2), eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion * t_total, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    effective_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
                                 (torch.distributed.get_world_size() if global_rank != -1 else 1)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", effective_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Save step every = %d", args.save_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=global_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        tr_batch_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=global_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  None if args.model_type in ['xlm', 'roberta'] else batch[2],
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':       batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_batch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                global_examples_seen = global_step * effective_train_batch_size

                epoch_iterator.desc = 'loss: {:.2e} lr: {:.2e}'.format(tr_batch_loss, scheduler.get_lr()[0])
                tr_batch_loss = 0.0
                if global_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_examples_seen)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_examples_seen)
                    logging_loss = tr_loss

                if global_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_examples_seen))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    # Evaluate model checkpoint (logging results)
                    if global_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, prefix=global_examples_seen)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_examples_seen)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if global_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    if args.preprocess_only:
        return

    write_dir = args.output_dir if args.write_dir is None else args.write_dir
    if not os.path.exists(write_dir) and global_rank in [-1, 0]:
        os.makedirs(write_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)  # if global_rank == -1 else DistributedSampler(dataset)  # No distributed eval to eval on full dev set
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type in ['xlm', 'roberta'] else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(write_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(write_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(write_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        all_predictions, all_nbest_predictions, all_null_odds = write_predictions_extended(
                        examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        all_predictions, all_nbest_predictions, all_null_odds = write_predictions(
                        examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    def filter_keys(qid_to_val, task_name):
        task_name = task_name.lower()
        assert task_name in {'hotpot', 'squad'}, 'task_name {} not implemented.'.format(task_name)
        return {qid: val for qid, val in qid_to_val.items() if len(qid.split('.')) == (2 if task_name == 'hotpot' else 1)}

    if len(filter_keys(all_predictions, 'squad')) == 0:
        results = {}  # No SQuAD data in evaluation set
    else:
        squad_output_prediction_file = os.path.join(write_dir, "squad_predictions_{}.json".format(prefix))
        with open(squad_output_prediction_file, 'w') as writer:
            writer.write(json.dumps(filter_keys(all_predictions, 'squad'), indent=2))
        squad_output_nbest_file = os.path.join(write_dir, "squad_nbest_predictions_{}.json".format(prefix))
        with open(squad_output_nbest_file, 'w') as writer:
            writer.write(json.dumps(filter_keys(all_nbest_predictions, 'squad'), indent=2))
        if args.version_2_with_negative:
            squad_output_null_log_odds_file = os.path.join(write_dir, "squad_null_odds_{}.json".format(prefix))
            with open(squad_output_null_log_odds_file, 'w') as writer:
                writer.write(json.dumps(filter_keys(all_null_odds, 'squad'), indent=2))
        else:
            squad_output_null_log_odds_file = None
        predict_file_parts = args.predict_file.split('/')
        squad_predict_file = '/'.join(predict_file_parts[:-2] + ['squad', predict_file_parts[-1]])
        evaluate_options = EVAL_OPTS(data_file=squad_predict_file,
                                     pred_file=squad_output_prediction_file,
                                     na_prob_file=squad_output_null_log_odds_file)
        results = evaluate_on_squad(evaluate_options)

    # Check if HotpotQA answer file exists to do HotpotQA evaluation
    hotpot_answer_file_parts = args.predict_file.split('/')
    hotpot_answer_file_parts[-2] = 'hotpot-orig'
    hotpot_answer_file = '/'.join(hotpot_answer_file_parts)
    if (not args.no_answer_file) and (not os.path.exists(hotpot_answer_file)):
        with open(os.path.join(write_dir, "squad_results_{}.json".format(prefix)), "w") as writer:
            writer.write(json.dumps(results, indent=2, sort_keys=True))
        return results

    # Evaluate with official HotpotQA script
    nbest_predictions = filter_keys(all_nbest_predictions, 'hotpot')
    null_odds = filter_keys(all_null_odds, 'hotpot')

    qids = {single_hop_qid.split('.')[0] for single_hop_qid in nbest_predictions.keys()}
    pred_answers_and_sps = {'answer': {}, 'sp': {}}
    globally_normed_pred_answers_and_sps = {'answer': {}, 'sp': {}}
    pred_infos = {}
    globally_normed_pred_infos = {}
    max_num_paragraphs = 10
    for qid in qids:
        # Find paragraph with answer prediction
        min_null_odds = float('inf')
        max_logit_sum = float('-inf')
        best_single_hop_qid = None
        for paragraph_no in range(max_num_paragraphs):
            single_hop_qid = qid + '.' + str(paragraph_no)
            if (single_hop_qid in null_odds) and (null_odds[single_hop_qid] < min_null_odds):
                best_single_hop_qid = single_hop_qid
                min_null_odds = null_odds[single_hop_qid]
            if single_hop_qid in nbest_predictions:
                for nbest_prediction in nbest_predictions[single_hop_qid]:
                    if (len(nbest_prediction['text']) > 0) and (args.model_type not in ['xlnet', 'xlm']):
                        logit_sum = nbest_prediction['start_logit'] + nbest_prediction['end_logit'] - null_odds[single_hop_qid]
                        if logit_sum > max_logit_sum:
                            globally_normed_pred_answers_and_sps['answer'][qid] = nbest_prediction['text']
                            globally_normed_pred_infos[qid] = nbest_prediction
                            max_logit_sum = logit_sum

        # Find/store answer and supporting fact
        pred_answers_and_sps['sp'][qid] = []  # NB: Dummy supporting fact for now
        globally_normed_pred_answers_and_sps['sp'][qid] = []  # NB: Dummy supporting fact for now
        for nbest_prediction in nbest_predictions[best_single_hop_qid]:
            if len(nbest_prediction['text']) > 0:
                pred_answers_and_sps['answer'][qid] = nbest_prediction['text']
                pred_infos[qid] = nbest_prediction
                break
        assert qid in pred_answers_and_sps['answer'], 'Error: No predicted answer found.'
        # assert qid in globally_normed_pred_answers_and_sps['answer'], 'Error: No globally normed predicted answer found.'

    hotpot_output_prediction_file = os.path.join(write_dir, "hotpot_predictions_{}.json".format(prefix))
    with open(hotpot_output_prediction_file, "w") as writer:
        writer.write(json.dumps(pred_answers_and_sps, indent=2))
    hotpot_results = evaluate_on_hotpot(hotpot_output_prediction_file, hotpot_answer_file) if not args.no_answer_file else {}
    with open(os.path.join(write_dir, "hotpot_predictions_info_{}.json".format(prefix)), "w") as writer:
        writer.write(json.dumps(pred_infos, indent=2))

    hotpot_output_prediction_gn_file = os.path.join(write_dir, "hotpot_predictions_gn_{}.json".format(prefix))
    with open(hotpot_output_prediction_gn_file, "w") as writer:
        writer.write(json.dumps(globally_normed_pred_answers_and_sps, indent=2))
    hotpot_gn_results = evaluate_on_hotpot(hotpot_output_prediction_gn_file, hotpot_answer_file) \
        if ((not args.no_answer_file) and (args.model_type not in ['xlnet', 'xlm'])) else {}
    with open(os.path.join(write_dir, "hotpot_predictions_gn_info_{}.json".format(prefix)), "w") as writer:
        writer.write(json.dumps(globally_normed_pred_infos, indent=2))

    hotpot_results = {k: v * 100. for k, v in hotpot_results.items()}
    hotpot_gn_results = {'gn_' + k: v * 100. for k, v in hotpot_gn_results.items()}
    results = {'squad_' + k: v for k, v in results.items()}
    results.update(hotpot_results)
    results.update(hotpot_gn_results)
    with open(os.path.join(write_dir, "hotpot_results_{}.json".format(prefix)), "w") as writer:
        writer.write(json.dumps(results, indent=2, sort_keys=True))
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()
    if global_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cache_filename = 'cached.split={}.mn={}.msl={}.mql={}'.format(
        '.'.join(input_file.split('/')[-1].split('.')[:-1]),
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        str(args.max_seq_length),
        str(args.max_query_length))
    if args.num_shards != 1:
        cache_filename += '.num_shards={}.shard_no={}'.format(args.num_shards, args.shard_no)
    cached_features_file = os.path.join(os.path.dirname(input_file), cache_filename)
    print('Cached features file {} exists: {}'.format(cached_features_file, str(os.path.exists(cached_features_file))))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        if args.num_shards == 1:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Loading features from cached files %s", cached_features_file)
            features = []
            for shard_no in tqdm(range(args.num_shards)):
                features += torch.load(cached_features_file[:-1] + str(shard_no))
        if output_examples:
            logger.info("Reading examples from file %s", input_file)
            examples = read_squad_examples(input_file=input_file,
                                                    is_training=not evaluate,
                                                    version_2_with_negative=args.version_2_with_negative)
    else:
        logger.info("Reading examples from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        logger.info("Creating features from dataset file at %s", input_file)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                num_shards=args.num_shards,
                                                shard_no=args.shard_no)
        if global_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--write_dir", default=None, type=str,
                        help="The write directory where evaluation/inference results will be saved.")
    parser.add_argument("--no_answer_file", action='store_true',
                        help="Set this flag if there is no answer file to evaluate with (i.e., just doing inference).")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=100, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=150, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--preprocess_only", action='store_true',
                        help="Whether to only run preprocessing.")
    parser.add_argument("--num_shards", default=1, type=int, help="Number of total shards to preprocess dataset with.")
    parser.add_argument("--shard_no", default=0, type=int, help="Shard number of dataset to preprocess.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=None, type=int,
                        help="Batch size per GPU/CPU for evaluation. Defaults to 2x --per_gpu_train_batch_size")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta_1", default=0.9, type=float,
                        help="Beta 1 for Adam optimizer.")
    parser.add_argument("--adam_beta_2", default=0.999, type=float,
                        help="Beta 2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument("--n_best_size", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_freq', type=int, default=1,
                        help="Save/evaluate X times per epoch. Overrides save_steps.")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Evaluate only this checkpoint directory within args.output_dir.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number."
                             "Overrides --eval_checkpoint")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--master_port", type=int, default=15349,
                        help="master_port for distributed training on gpus. Must be unused and in [12000, 20000]")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    if args.per_gpu_eval_batch_size is None:
        args.per_gpu_eval_batch_size = args.per_gpu_train_batch_size * 2
    assert args.shard_no < args.num_shards, '--shard_no {} should not exceed --num_shards {}'.format(args.shard_no, args.num_shards)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    print('-' * 40, 'START', '-' * 40, args.local_rank)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        global_rank = -1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_PORT'] = str(args.master_port)
        print('MASTER_PORT:', os.environ['MASTER_PORT'])
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        global_rank = torch.distributed.get_rank()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if global_rank in [-1, 0] else logging.WARN)
    logger.warning("Process local rank: %s, global rank %s/%s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, global_rank, (torch.distributed.get_world_size() if global_rank != -1 else -1), device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if global_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if global_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Set up tensorboard metric logging
    tb_writer = SummaryWriter(args.output_dir) if global_rank in [-1, 0] else None

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        if not args.preprocess_only:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, tb_writer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # Save the trained model and the tokenizer
            if global_rank in [-1, 0]:
                # Create output directory if needed
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

                # Load a trained model and vocabulary that you have fine-tuned
                model = model_class.from_pretrained(args.output_dir)
                tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
                model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if (global_rank == 0) or (args.do_eval and global_rank == -1):
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        elif args.eval_checkpoint is not None:
            checkpoints = [os.path.join(args.output_dir, args.eval_checkpoint)]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_examples_seen = checkpoint.split('-')[-1] if ((len(checkpoints) > 1) or (args.eval_checkpoint is not None)) else \
                args.predict_file.split('/')[-1].rsplit('.', 1)[0]
            if not args.preprocess_only:  # Use pre-trained model/tokenizer loaded earlier
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_examples_seen)
            if args.preprocess_only:
                return

            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_examples_seen)
            result = dict((k + ('_{}'.format(global_examples_seen) if global_examples_seen else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
