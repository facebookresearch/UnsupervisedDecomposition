# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import random


DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'XLM/data')


def main():
    split = 'dev'
    data_file = f'{DATA_DIR}/hotpot-orig/{split}.json'
    with open(data_file) as f:
        exs = json.load(f)

    print('Randomly splitting examples...')
    random.Random(42).shuffle(exs)
    exs1, exs2 = exs[:len(exs)//2], exs[len(exs)//2:]

    with open(f'{DATA_DIR}/hotpot-orig/{split}1.json', 'w') as f:
        json.dump(exs1, f, indent=2)
    with open(f'{DATA_DIR}/hotpot-orig/{split}1.qids.txt', 'w') as f:
        f.writelines('\n'.join([ex['_id'] for ex in exs1]))
    with open(f'{DATA_DIR}/hotpot-orig/{split}1.mh', 'w') as f:
        f.writelines('\n'.join([ex['question'].strip() for ex in exs1]) + '\n')
    print(f'Saved to: {DATA_DIR}/hotpot-orig/{split}1.json')
    with open(f'{DATA_DIR}/hotpot-orig/{split}2.json', 'w') as f:
        json.dump(exs2, f, indent=2)
    with open(f'{DATA_DIR}/hotpot-orig/{split}2.qids.txt', 'w') as f:
        f.writelines('\n'.join([ex['_id'] for ex in exs2]))
    with open(f'{DATA_DIR}/hotpot-orig/{split}2.mh', 'w') as f:
        f.writelines('\n'.join([ex['question'].strip() for ex in exs2]) + '\n')
    print(f'Saved to: {DATA_DIR}/hotpot-orig/{split}2.json')


if __name__ == "__main__":
    main()
