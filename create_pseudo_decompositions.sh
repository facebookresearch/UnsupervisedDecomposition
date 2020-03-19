# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --main_dir)
    MAIN_DIR="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


# Make pseudo-decomposition data and replace single-hop question entities with multi-hop question entities
cd $MAIN_DIR/XLM
DATA_FOLDER=all.pseudo_decomp  # Should start with "all.", since we'll start by using all training and validation examples in HotpotQA
for SPLIT in dev train; do
    python ../pytorch-transformers/pseudoalignment/pseudo_decomp_fasttext.py --split $SPLIT --data_folder data/umt/$DATA_FOLDER --beam_size 1000
done
for SPLIT in valid train; do
    python ../pytorch-transformers/pseudoalignment/replace_subq_entities.py --split $SPLIT --data_folder $DATA_FOLDER --replace_by_type
done
DATA_FOLDER=$DATA_FOLDER.replace_entity_by_type

# Copy over the original multi-hop questions (since we always train on unchanged HotpotQA questions)
cp data/umt/all/train.mh data/umt/$DATA_FOLDER/
cp data/umt/all/valid.mh data/umt/$DATA_FOLDER/

# Use training examples as test data to track training loss
tail -5000 data/umt/$DATA_FOLDER/train.mh > data/umt/$DATA_FOLDER/test.mh
tail -5000 data/umt/$DATA_FOLDER/train.sh > data/umt/$DATA_FOLDER/test.sh

# Randomly half the dev set into dev1 and dev2, and make/use a version of the data which only uses dev1 for validation
python ../pytorch-transformers/split_umt_dev.py --data_folder $DATA_FOLDER
DATA_FOLDER=dev1.${DATA_FOLDER:4}

# Preprocess data
./get-data-mt.sh --src mh --tgt sh --reload_codes dumped/xlm_en/codes_en --reload_vocab dumped/xlm_en/vocab_en --data_folder $DATA_FOLDER
