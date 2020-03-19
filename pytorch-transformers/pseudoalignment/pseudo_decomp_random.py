import argparse
import json
import os
import random
from tqdm import tqdm
from pseudo_decomp_utils import get_squad_path


def read_lines_from_path(path):
    """
    Utility to read stripped lines from specified filepath
    """
    with open(path) as f:
        lines = f.readlines()
    return [line.strip('\n').strip() for line in lines]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='dev', choices=['train', 'dev', 'valid'], type=str,
                        help="Pair Qs for which split?")
    parser.add_argument("--data_folder", type=str, help="Path to save results to")
    args = parser.parse_args()
    args.umt_data_folder = 'squad'

    print('Loading questions...')
    if args.umt_data_folder == 'squad':
        with open(get_squad_path(args.split)) as f:
            data = json.load(f)

        qs = []
        for article in tqdm(data['data']):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qs.append(qa['question'].strip('\n').strip())
    else:
        qs = read_lines_from_path(f'{args.umt_data_folder}/{args.split}.sh1')

    print('Pairing questions...')
    qs_shuf = qs.copy()
    random.Random(42).shuffle(qs_shuf)
    q_pairs = list(zip(qs, qs_shuf))
    q_pairs = [' '.join(q_pair) for q_pair in q_pairs]

    print('Saving to file...')
    save_filepath = f'{args.data_folder}/{"valid" if args.split == "dev" else args.split}.sh'
    os.makedirs(args.data_folder, exist_ok=True)
    with open(save_filepath, 'w') as f:
        f.writelines('\n'.join(q_pairs) + '\n')
    print(f'Done! Saved to {save_filepath}')


if __name__ == "__main__":
    main()
