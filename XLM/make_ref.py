# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | python make_ref.py --output_path $OUTPUT_PATH
#

import os
import sys
import argparse

from src.utils import restore_segmentation


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = argparse.ArgumentParser(description="Generate reference file")
    parser.add_argument("--output_path", type=str, default="", help="Output reference path")
    params = parser.parse_args()
    assert params.output_path and not os.path.isfile(params.output_path)

    # read sentences from stdin
    src_sent = []
    for line in sys.stdin.readlines():
        assert len(line.strip().split()) > 0
        src_sent.append(line.strip().replace('<unk>', '<<unk>>'))
    print("Read %i sentences from stdin." % len(src_sent))

    # export sentences to file / restore BPE segmentation
    with open(params.output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src_sent) + '\n')
    restore_segmentation(params.output_path)
    print("Restored segmentation")
