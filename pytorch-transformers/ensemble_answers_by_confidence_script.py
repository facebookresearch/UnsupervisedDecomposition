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
from examples.hotpot_evaluate_v1 import eval as evaluate_on_hotpot
from scipy.special import softmax
from tqdm import tqdm

DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'pytorch-transformers/data')
CUR_DIR = os.path.join(os.environ['MAIN_DIR'], 'pytorch-transformers')


def main():
    # Ensembling RoBERTA models (need to re-run eval for preds_file1): python ensemble_answers_by_confidence_script.py --preds_file1 checkpoint/v8.tn=hotpot-all.umt.all.model=20639223.st=0.0.beam=5.lp=1.0.seed=0.atype=sentence-1-center.use_q.use_suba.suba1=0.suba2=0-squad.medium_hard_frac=1.0.mn=roberta-large.bs=64.lr=1.5e-5.nte=2.wd=0.01.rs={}/nbest_predictions_dev.json --preds_file2 checkpoint/v4.tn=hotpot-squad.mn=roberta-large.bs=64.lr=1.5e-5.nte=2.wd=0.01.rs={}/nbest_predictions_dev.json --seeds 5
    # python ensemble_answers_by_confidence_script.py --preds_file1 checkpoint/v8.tn=hotpot-all.umt.all.model=20639223.st=0.0.beam=5.lp=1.0.seed=0.q-predsubqs-predsubas-sentence.suba1=0.suba2=0-squad.medium_hard_frac=1.0.mn=roberta-large.bs=64.lr=1.5e-5.nte=2.wd=0.01.rs={}/nbest_predictions_dev.json --preds_file2 checkpoint/v8.tn=hotpot-all.umt.all.model=20639223.st=0.0.beam=5.lp=1.0.seed=0.atype=sentence-1-center.use_q.use_suba.suba1=0.suba2=0-squad.medium_hard_frac=1.0.mn=roberta-large.bs=64.lr=1.5e-5.nte=2.wd=0.01.rs={}/nbest_predictions_dev.json --seeds 5
    # python ensemble_answers_by_confidence_script.py --seeds 5 --no_answer_file --split dev --preds_file1 data/hotpot.umt.all.model=20639223.st=0.0.beam=5.lp=1.0.seed=0/roberta_predict.rs={}/nbest_predictions_dev.json
    # python ensemble_answers_by_confidence_script.py --seeds 5 --no_answer_file --split train --preds_file1 data/hotpot.umt.all.model=20639223.st=0.0.beam=5.lp=1.0.seed=0/roberta_predict.rs={}/nbest_predictions_train.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file1", type=str, required=True,  # E.g., 'checkpoint/v4.tn=hotpot-squad.mn=bert-large-uncased-whole-word-masking.bs=32.lr=2e-5.rs={}.ngpu=8/nbest_predictions_.json'
                        help="Prediction files to ensemble (from model 1). Will vary random seed if path contains {}.")
    parser.add_argument("--preds_file2", type=str, default=None,  # E.g., 'checkpoint/v4.tn=hotpot-all.umt.all.model=20464567.st=0.0.beam=5.lp=1.0.seed=0.q-predsubqs-predsubas-nobadsubqs.suba1=0.suba2=0-squad.medium_hard_frac=1.0.mn=bert-large-uncased-whole-word-masking.lr=2e-5.rs={}/nbest_predictions_.json'
                        help="Prediction files to ensemble (from model 2). Will vary random seed if path contains {}.")
    parser.add_argument("--split", default='dev', type=str,
                        help="HotpotQA Split to evaluate on, e.g. 'train', 'dev', 'hotpot_dev_distractor_adversarial_jiang_v1'")
    parser.add_argument("--seeds", default=4, type=int,
                        help="Number of random seeds. Ignore if using --seeds_list")
    parser.add_argument('--seeds_list', nargs='+', type=int, default=None,
                        help='List of specific random seed numbers to use. Overrides --seeds')
    parser.add_argument("--no_answer_file", action='store_true', default=False,
                        help="No gold answer file to evaluate with")
    args = parser.parse_args()
    args.metrics = ['gn_f1', 'gn_bridge_f1', 'gn_comparison_f1', 'gn_onehop_f1', 'gn_multihop_f1']
    seeds = [i for i in range(args.seeds)] if args.seeds_list is None else args.seeds_list
    seeds_str = "-".join([str(i) for i in seeds])

    print('Ensembling All Model 1')
    preds_files1 = [args.preds_file1.format(i) for i in seeds] if '{}' in args.preds_file1 else [args.preds_file1]
    ensembled_file1 = args.preds_file1.replace('{}', seeds_str)
    evaluate_ensemble(preds_files1, ensembled_file1, args)
    if args.preds_file2 is None:
        return

    print('Ensembling All Model 2')
    preds_files2 = [args.preds_file2.format(i) for i in seeds] if '{}' in args.preds_file2 else [args.preds_file2]
    ensembled_file2 = args.preds_file2.replace('{}', seeds_str)
    evaluate_ensemble(preds_files2, ensembled_file2, args)

    print('Ensembling Model 1 + Model 2')
    ensemble_all_pairs(preds_files1, preds_files2, args)
    print('Ensembling Model 1 + Model 1')
    ensemble_all_pairs(preds_files1, preds_files1, args)
    print('Ensembling Model 2 + Model 2')
    ensemble_all_pairs(preds_files2, preds_files2, args)


def ensemble_all_pairs(preds_files1, preds_files2, args):
    ensemble_results = []
    for preds_file1 in preds_files1:
        for preds_file2 in preds_files2:
            if preds_file1 == preds_file2:
                continue
            ensemble_results.append(evaluate_ensemble([preds_file1, preds_file2], args))
    if len(ensemble_results) > 0:
        for metric in args.metrics:
            values = np.array([r[metric] for r in ensemble_results])
            print('{:.20s}: {:.2f} +/- {:.2f} // {:.2f} - {:.2f}'.format(metric + 20 * ' ', values.mean(), round(values.std(), 2), round(values.min(), 2), round(values.max(), 2)))


def evaluate_ensemble(filepaths, ensembled_file, args):
    # Write ensembled nbest_predictions.json
    prefix = f'{args.split}'
    hotpot_answer_file = f'{DATA_DIR}/hotpot-orig/{args.split}.json'
    model_weights = np.ones(len(filepaths)) / len(filepaths)
    all_nbest_predictions = []
    all_null_odds = []
    print('    Reading files...')
    for filepath in tqdm(filepaths):
        with open(f'{CUR_DIR}/{filepath}') as f:
            all_nbest_predictions.append(json.load(f))
        with open(f'{CUR_DIR}/{filepath}'.replace("nbest_predictions", "null_odds")) as f:
            all_null_odds.append(json.load(f))

    qids = all_nbest_predictions[0].keys()
    ensembled_nbest_predictions = {}
    ensembled_null_odds = {}
    print('    Ensembling...')
    for qid in tqdm(qids):
        text2prob, text2start_logit, text2end_logit = {}, {}, {}
        for model_no, (nbest_prediction, null_odds) in enumerate(zip(all_nbest_predictions, all_null_odds)):
            for pred in nbest_prediction[qid]:
                text_evidence = (pred['text'], pred.get('evidence', ''))
                text2prob[text_evidence] = text2prob.get(text_evidence, 0.) + (model_weights[model_no] * pred['probability'])
                text2start_logit[text_evidence] = text2start_logit.get(text_evidence, 0.) + (model_weights[model_no] * pred['start_logit'])
                text2end_logit[text_evidence] = text2end_logit.get(text_evidence, 0.) + (model_weights[model_no] * pred['end_logit'])
            ensembled_null_odds[qid] = ensembled_null_odds.get(qid, 0.) + (model_weights[model_no] * null_odds[qid])
        ensembled_nbest_predictions[qid] = []
        for text_evidence in sorted(text2prob, key=text2prob.get, reverse=True):
            ensembled_nbest_predictions[qid].append({
                'text': text_evidence[0],
                'evidence': text_evidence[1],
                'probability': text2prob[text_evidence],
                'start_logit': text2start_logit[text_evidence],
                'end_logit': text2end_logit[text_evidence],
                'logit': text2start_logit[text_evidence] + text2end_logit[text_evidence] - ensembled_null_odds[qid],
                'no_answer_logit': ensembled_null_odds[qid],
            })
    ensembled_dir = ensembled_file.rsplit('/', 1)[0]
    os.makedirs(ensembled_dir, exist_ok=True)
    ensembled_filepath = os.path.join(ensembled_dir, f'nbest_predictions_{prefix}.json')
    print(f'    Saving ensemble predictions...')
    with open(f'{CUR_DIR}/{ensembled_filepath}', 'w') as f:
        json.dump(ensembled_nbest_predictions, f, indent=2)
    with open(f'{CUR_DIR}/{ensembled_filepath}'.replace("nbest_predictions", "null_odds"), 'w') as f:
        json.dump(ensembled_null_odds, f, indent=2)

    all_best_paragraph_probabilities = []
    best_paragraph_probabilities = []
    # Load answer predictions and confidences
    output_dir = f'{CUR_DIR}/{ensembled_dir}'
    with open(f'{CUR_DIR}/{ensembled_filepath}') as f:
        nbest_predictions = json.load(f)
    with open(f'{CUR_DIR}/{ensembled_filepath}'.replace('nbest_predictions', 'null_odds')) as f:
        null_odds = json.load(f)
    qids = {paraqid.split('.')[0] for paraqid in nbest_predictions.keys() if '.' in paraqid}
    print('# of eval Qs:', len(qids))

    # Get predicted answers
    pred_answers_and_sps = {'answer': {}, 'sp': {}, 'probability': {}, 'start_logit': {}, 'end_logit': {}}
    globally_normed_pred_answers_and_sps = {'answer': {}, 'sp': {}, 'probability': {}, 'start_logit': {}, 'end_logit': {}}
    pred_infos = {}
    globally_normed_pred_infos = {}
    max_num_paragraphs = 10
    for qid in qids:
        # Find paragraph with answer prediction
        min_null_odds = float('inf')
        max_logit_sum = float('-inf')
        best_single_hop_qid = None
        example_null_odds = []
        best_paragraph_no = 0
        for paragraph_no in range(max_num_paragraphs):
            single_hop_qid = qid + '.' + str(paragraph_no)
            if single_hop_qid in null_odds:
                example_null_odds.append(null_odds[single_hop_qid])
                if null_odds[single_hop_qid] < min_null_odds:
                    best_single_hop_qid = single_hop_qid
                    best_paragraph_no = paragraph_no
                    min_null_odds = null_odds[single_hop_qid]
            if single_hop_qid in nbest_predictions:
                for nbest_prediction in nbest_predictions[single_hop_qid]:
                    if len(nbest_prediction['text']) > 0:
                        logit_sum = nbest_prediction['start_logit'] + nbest_prediction['end_logit'] - null_odds[single_hop_qid]
                        if logit_sum > max_logit_sum:
                            globally_normed_pred_answers_and_sps['answer'][qid] = nbest_prediction['text']
                            globally_normed_pred_infos[qid] = nbest_prediction
                            for key in ['probability', 'start_logit', 'end_logit']:
                                globally_normed_pred_answers_and_sps[key][qid] = nbest_prediction[key]
                            max_logit_sum = logit_sum
        paragraph_logits = -np.array(example_null_odds)
        best_paragraph_probability = softmax(paragraph_logits)[best_paragraph_no]
        best_paragraph_probabilities.append(best_paragraph_probability)

        # Find/store answer and supporting fact
        pred_answers_and_sps['sp'][qid] = []  # NB: Dummy supporting fact for now
        globally_normed_pred_answers_and_sps['sp'][qid] = []  # NB: Dummy supporting fact for now
        for nbest_prediction in nbest_predictions[best_single_hop_qid]:
            if len(nbest_prediction['text']) > 0:
                pred_answers_and_sps['answer'][qid] = nbest_prediction['text']
                pred_infos[qid] = nbest_prediction
                for key in ['probability', 'start_logit', 'end_logit']:
                    pred_answers_and_sps[key][qid] = nbest_prediction[key]
                # pred_answers_and_sps['probability'][qid] *= best_paragraph_probability
                break
        assert qid in pred_answers_and_sps['answer'], 'Error: No predicted answer found.'
        assert qid in globally_normed_pred_answers_and_sps['answer'], 'Error: No globally normed predicted answer found.'

    hotpot_output_prediction_gn_file = os.path.join(output_dir, "hotpot_predictions_gn_{}.json".format(prefix))
    with open(hotpot_output_prediction_gn_file, "w") as writer:
        writer.write(json.dumps(globally_normed_pred_answers_and_sps, indent=2))
    hotpot_gn_results = evaluate_on_hotpot(hotpot_output_prediction_gn_file, hotpot_answer_file) if not args.no_answer_file else {}
    with open(os.path.join(output_dir, "hotpot_predictions_gn_info_{}.json".format(prefix)), "w") as writer:
        writer.write(json.dumps(globally_normed_pred_infos, indent=2))
    print(f'    Saved to {os.path.join(output_dir, "hotpot_predictions_gn_info_{}.json".format(prefix))}')

    hotpot_gn_results = {'gn_' + k: v * 100. for k, v in hotpot_gn_results.items()}
    all_best_paragraph_probabilities.append(np.array(best_paragraph_probabilities))
    return hotpot_gn_results


if __name__ == "__main__":
    main()
