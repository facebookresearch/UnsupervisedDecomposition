# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import shutil


DATA_DIR = os.path.join(os.environ["MAIN_DIR"], 'XLM/data')


def read_lines(path):
    """
    Utility to read stripped lines from specified filepath
    """
    with open(path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def write_lines(lines, path):
    """
    Utility to write lines from specified filepath
    """
    with open(path, 'w') as f:
        f.writelines('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help="Name of UMT data directory, e.g. all.paired_beam...")
    args = parser.parse_args()
    prefix = 'all'
    assert args.data_folder.startswith(prefix)

    old_data_path = f'{DATA_DIR}/umt/{args.data_folder}'
    new_data_folder = 'dev1' + args.data_folder[len(prefix):]
    new_data_path = f'{DATA_DIR}/umt/{new_data_folder}'

    # Copy over train and test (slice of train)
    print(f'Copying train/test to: {new_data_path}')
    os.makedirs(new_data_path, exist_ok=True)
    for filename in ['train.mh', 'train.sh', 'test.mh', 'test.sh']:
        shutil.copyfile(os.path.join(old_data_path, filename), os.path.join(new_data_path, filename))

    # Copy over half-subset of validation
    print(f'Copying validation subset to: {new_data_path}')
    valid_mh = read_lines(f'{old_data_path}/valid.mh')
    valid_sh = read_lines(f'{old_data_path}/valid.sh')
    valid_qids = read_lines(f'{DATA_DIR}/umt/all/valid.qids.txt')
    valid1_qids = set(read_lines(f'{DATA_DIR}/hotpot-orig/dev1.qids.txt'))
    valid1_mh, valid1_sh, valid1_qids = zip(*[(mh, sh, qid) for mh, sh, qid in zip(valid_mh, valid_sh, valid_qids) if qid in valid1_qids])
    write_lines(valid1_mh, f'{new_data_path}/valid.mh')
    write_lines(valid1_sh, f'{new_data_path}/valid.sh')
    write_lines(valid1_qids, f'{new_data_path}/valid.qids.txt')
    print(f'Done! Saved to: {new_data_path}')


if __name__ == "__main__":
    main()
