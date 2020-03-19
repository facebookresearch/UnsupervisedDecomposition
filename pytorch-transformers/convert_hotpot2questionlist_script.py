# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, help="HotpotQA formatted file to convert to SQuAD format, e.g. test1")
    args = parser.parse_args()

    # Post-process args
    if args.split.endswith('.json'):
        args.split = args.split.rsplit('.json')[0]
    args.filename = args.split + '.json'

    read_filename = f'data/hotpot-orig/{args.filename}'
    print('Reading from:', read_filename)
    with open(read_filename, 'r') as f:
        data_hotpot_split = json.load(f)

    save_split = args.split.replace('dev', 'valid')
    save_folder = f'data/umt/all'
    os.makedirs(save_folder, exist_ok=True)
    with open(f'{save_folder}/{save_split}.mh', 'w') as f:
        f.writelines('\n'.join([ex['question'].strip() for ex in data_hotpot_split]) + '\n')
    print('Saved to:', f'{save_folder}/{save_split}.mh')

    with open(f'{save_folder}/{save_split}.qids.txt', 'w') as f:
        f.writelines('\n'.join([ex['_id'] for ex in data_hotpot_split]))
    print('Saved to:', f'{save_folder}/{save_split}.qids.txt')


if __name__ == '__main__':
    main()
