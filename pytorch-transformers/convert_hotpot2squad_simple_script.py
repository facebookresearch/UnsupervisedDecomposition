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
from pprint import pprint
from tqdm import tqdm


def convert_example(example, bad_answer_example=False, use_title=False):
    yes_q = example['answer'].lower().strip() == 'yes'
    no_q = example['answer'].lower().strip() == 'no'
    span_q = (not yes_q) and (not no_q)
    clean_answer = example['answer'].strip()
    if bad_answer_example:
        clean_answer = clean_answer.strip('"')
        clean_answer = clean_answer.replace('3OH!3', '3OH! 3')  # Hard-coded special case to match text
    num_answerable = 0
    supporting_fact_titles = set([sf[0] for sf in example['supporting_facts']])
    new_example = {"title": example['question'], 'paragraphs': []}
    found_sfs = 0
    for paragraph_index, (title, sents) in enumerate(example['context']):
        new_context_start_tokens = ['yes', 'no']
        if use_title:
            new_context_start_tokens += [title.strip(), '/']
        new_context = ' '.join(new_context_start_tokens)

        answer_start = None
        if title in supporting_fact_titles:
            if yes_q:
                answer_start = new_context.index('yes')
            elif no_q:
                answer_start = new_context.index('no')

        supporting_facts = []
        for sent_index, sent in enumerate(sents):
            clean_sent = sent.strip()
            if [title, sent_index] in example['supporting_facts']:
                supporting_facts.append(clean_sent)  # Found supporting fact
                if span_q and (title in supporting_fact_titles) and (answer_start is None) and (clean_answer in clean_sent):  # Find span if possible (Only use 1st span)
                    answer_start = len(new_context + ' ') + clean_sent.index(clean_answer)
            new_context += ' ' + clean_sent

        if span_q and (title in supporting_fact_titles) and (answer_start is None) and (clean_answer in new_context):
            answer_start = new_context.index(clean_answer)

        new_example['paragraphs'].append({
            # HotpotQA info
            "_id": example['_id'],
            "type": example['type'],
            "level": example['level'],
            "supporting_facts": supporting_facts,
            # SQuAD info
            "context": new_context,
            "qas": [
                {
                    "question": example['question'],
                    "id": example['_id'] + '.' + str(paragraph_index),
                    "answers": [{
                        "text": clean_answer,
                        "answer_start": answer_start
                    }] if answer_start is not None else [],
                    "is_impossible": answer_start is None
                }
            ]
        })

        # Verification
        if answer_start is not None:
            num_answerable += 1
            assert clean_answer == new_context[answer_start: answer_start + len(clean_answer)], \
                '[{}] answer {} != span {}'.format(
                example['_id'] + '.' + str(paragraph_index),
                clean_answer,
                new_context[answer_start: answer_start + len(clean_answer)])
            assert title in supporting_fact_titles, 'title {} not in supporting fact titles: {} {} for new example {}'.format(
                title, supporting_fact_titles, example['supporting_facts'], new_example['paragraphs'][-1])
        found_sfs += len(supporting_facts)
    missing_sfs = len(example['supporting_facts']) - found_sfs
    return new_example, num_answerable, missing_sfs, found_sfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, help="HotpotQA formatted file to convert to SQuAD format, e.g. hotpot_train_v1")
    parser.add_argument('--context_is_lower', action='store_true', help="Is the context lower cased? (e.g., for hotpot_dev_distractor_adversarial_min_v1.json)")
    args = parser.parse_args()

    # Post-process args
    if args.split.endswith('.json'):
        args.split = args.split.rsplit('.json')[0]
    args.filename = args.split + '.json'
    if args.filename == 'hotpot_dev_distractor_adversarial_min_v1.json':
        print('Context is lower-cased for ' + args.filename + '. Setting args.context_is_lower = True')
        args.context_is_lower = True

    print('Converting file:', args.filename)
    new_data_dir = 'data/hotpot'
    os.makedirs(new_data_dir, exist_ok=True)
    read_filename = 'data/hotpot-orig/{}'.format(args.filename)
    print('Reading from:', read_filename)
    with open(read_filename, 'r') as f:
        data_hotpot_split = json.load(f)

    new_data = {'data': [], 'version': args.filename}
    num_answerable_per_q = []
    num_bad_answer_examples = 0
    total_missing_sfs = 0
    total_found_sfs = 0
    num_answer_words = []
    num_question_words = []
    for example_index, example in enumerate(tqdm(data_hotpot_split)):
        if args.context_is_lower:
            example['answer'] = example['answer'].lower()
            for i in range(len(example['supporting_facts'])):
                example['supporting_facts'][i][0] = example['supporting_facts'][i][0].lower().replace('–', '-').replace('&', 'and').replace('&amp;', 'and')
        new_example, num_answerable, missing_sfs, found_sfs = convert_example(example)
        if num_answerable == 0:
            num_bad_answer_examples += 1
            new_example, num_answerable, missing_sfs, found_sfs = convert_example(example, bad_answer_example=True)
            if num_answerable == 0:
                pprint(example)
        new_data['data'].append(new_example)
        num_answerable_per_q.append(num_answerable)
        total_missing_sfs += missing_sfs
        total_found_sfs += found_sfs
        num_answer_words.append(len(example['answer'].strip().split()))
        num_question_words.append(len(example['question'].strip().split()))

    num_answerable_per_q = np.array(num_answerable_per_q)
    print('{}: # Bad Answers: {}'.format(args.filename, num_bad_answer_examples))
    print('{}: # Unanswerable: {}'.format(args.filename, (num_answerable_per_q == 0).sum()))
    print('{}: # Missing SFs: {}'.format(args.filename, total_missing_sfs))
    print('{}: # Found SFs: {}'.format(args.filename, total_found_sfs))

    save_filename = os.path.join(new_data_dir, args.filename)
    with open(save_filename, 'w') as f:
        json.dump(new_data, f, indent=2, sort_keys=False)
    print('Saved to:', save_filename)


if __name__ == '__main__':
    main()
