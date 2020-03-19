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


# Make directories to save data and models, and symlink these together
cd $MAIN_DIR/XLM
mkdir -p checkpoint dumped data/umt
cd $MAIN_DIR/pytorch-transformers
ln -s ../XLM/checkpoint checkpoint
ln -s ../XLM/dumped dumped
ln -s ../XLM/data data

# Download XLM's English-pretrained model vocab and BPE codes (we'll use these for pre-processing)
cd $MAIN_DIR/XLM
mkdir -p dumped/xlm_en
wget https://dl.fbaipublicfiles.com/XLM/codes_en
mv codes_en dumped/xlm_en/
wget https://dl.fbaipublicfiles.com/XLM/vocab_en
mv vocab_en dumped/xlm_en/

# Download HotpotQA data using links from https://hotpotqa.github.io/
mkdir -p data/hotpot-orig  # Make folder for original HotpotQA data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
mv hotpot_train_v1.1.json data/hotpot-orig/hotpot_train_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
mv hotpot_dev_distractor_v1.json data/hotpot-orig/
# Sometimes, we reference files using different names/paths
cd data/hotpot-orig
ln -s hotpot_train_v1.json train.json
ln -s hotpot_dev_distractor_v1.json dev.json

# Write HotpotQA questions and QIDs to XLM-formatted questions file
cd $MAIN_DIR/XLM/
for SPLIT in dev train; do
    python ../pytorch-transformers/convert_hotpot2questionlist_script.py --split $SPLIT
    python ../pytorch-transformers/convert_hotpot2squad_simple_script.py --split $SPLIT
done

for SPLIT in valid train; do
    ./get-data-mt-split-src-only.sh --src mh --tgt sh --reload_codes dumped/xlm_en/codes_en --reload_vocab dumped/xlm_en/vocab_en --data_folder all --split $SPLIT
done

# Download SQuAD data to create pseudo-decompositions
mkdir -p data/squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
mv train-v2.0.json data/squad
mv dev-v2.0.json data/squad

# Copy pretrained MLM model
mkdir -p $MAIN_DIR/XLM/dumped/mlm.dev1.pseudo_decomp_random.mined
wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/dumped/mlm.dev1.pseudo_decomp_random.mined/best-valid_mlm_ppl.pth
mv best-valid_mlm_ppl.pth dumped/mlm.dev1.pseudo_decomp_random.mined/
# Copy pre-trained decomposition model
mkdir -p dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223
for FILE in best-valid_mh-sh-mh_mt_effective_goods_back_bleu.pth hyp.st=0.0.bs=5.lp=1.0.es=False.seed=0.mh-sh.train.pred.bleu.sh.txt hyp.st=0.0.bs=5.lp=1.0.es=False.seed=0.mh-sh.valid.pred.bleu.sh.txt; do
    wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/$FILE
    mv $FILE dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/
done
# Copy single-hop question answering model ensemble
for NUM_PARAGRAPHS in 1 3; do
    mkdir -p checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/
    for FILE in config.json pytorch_model.bin training_args.bin; do
        wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/$FILE
        mv $FILE checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/
    done
done
# Copy bridge/comparison/intersection/onehop splits, plus Jiang et al.'s adversarial multi-hop dev set and Min et al.'s out-of-domain dev set
for FILE in dev.bridge.include_onehop.qids.txt dev.comparison.include_onehop.qids.txt dev.intersec.include_onehop.qids.txt dev.onehop.include_onehop.qids.txt hotpot_dev_distractor_adversarial_jiang_v1.json hotpot_dev_distractor_adversarial_min_cased_with_sentence_seg_v1.json; do
    wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/data/hotpot-orig/$FILE
    mv $FILE $MAIN_DIR/XLM/data/hotpot-orig/
done

# Reproduce our held-in/held-out dev splits
python ../pytorch-transformers/split_hotpot_dev.py

# Download FastText vectors for creating pseudo-decompositions
mkdir -p data/fastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec data/fastText/
rm crawl-300d-2M.vec.zip
cd $MAIN_DIR
