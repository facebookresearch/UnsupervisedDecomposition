import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

def read_lines(path):
    """
    Utility to read stripped lines from specified filepath
    """
    with open(path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total'] += 1
    return em, prec, recall, f1

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file, verbose=True):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    with open('/'.join(gold_file.split('/')[:-1] + ['hotpot_dev_1hop_solv_nonsolv.json'])) as f:
        onehop_ids = set(json.load(f)['1hop_solv'])
    dev1_ids = set(read_lines('/'.join(gold_file.split('/')[:-1] + ['dev1.qids.txt'])))
    b_ids = set(read_lines('/'.join(gold_file.split('/')[:-1] + ['dev.bridge.include_onehop.qids.txt'])))
    c_ids = set(read_lines('/'.join(gold_file.split('/')[:-1] + ['dev.comparison.include_onehop.qids.txt'])))
    i_ids = set(read_lines('/'.join(gold_file.split('/')[:-1] + ['dev.intersec.include_onehop.qids.txt'])))

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'total': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    subset_metrics = {subset: {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'total': 0}
                      for subset in ['bridge', 'comparison', 'onehop', 'multihop', 'b', 'c', 'i', 'o',
                                     'dev1', 'dev1_bridge', 'dev1_comparison', 'dev1_onehop', 'dev1_multihop', 'dev1_b', 'dev1_c', 'dev1_i', 'dev1_o',
                                     'dev2', 'dev2_bridge', 'dev2_comparison', 'dev2_onehop', 'dev2_multihop', 'dev2_b', 'dev2_c', 'dev2_i', 'dev2_o']}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall, f1 = update_answer(metrics, prediction['answer'][cur_id], dp['answer'])
            num_hops = 'onehop' if cur_id in onehop_ids else 'multihop'
            dev_split = 'dev1' if cur_id in dev1_ids else 'dev2'
            type_split = 'i' if cur_id in i_ids else ('c' if cur_id in c_ids else ('b' if cur_id in b_ids else 'o'))
            _ = update_answer(subset_metrics[dp['type']], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[num_hops], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[type_split], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[dev_split], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[dev_split + '_' + dp['type']], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[dev_split + '_' + num_hops], prediction['answer'][cur_id], dp['answer'])
            _ = update_answer(subset_metrics[dev_split + '_' + type_split], prediction['answer'][cur_id], dp['answer'])

        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N
    metrics.pop('total')
    for subset in subset_metrics.keys():
        for k in subset_metrics[subset].keys():
            if (k != 'total') and (subset_metrics[subset]['total'] != 0):
                metrics[subset + '_' + k] = subset_metrics[subset][k] / subset_metrics[subset]['total']

    if verbose:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])
