# Usage: ./get-data-umt.sh --src mh --tgt sh --reload_codes dumped/xlm_en/codes_en --reload_vocab dumped/xlm_en/vocab_en --data_folder all
# This script will successively:
# 1) download Moses scripts, download and compile fastBPE
# 2) tokenize and apply BPE to monolingual and parallel test data
# 3) binarize all datasets

set -e


#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --reload_codes)
    RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
    RELOAD_VOCAB="$2"; shift 2;;
  --data_folder)
    DATA_FOLDER="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

#
# Check parameters
#
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$SRC" != "mh" -a "$SRC" != "sh" ]; then echo "unknown source language"; exit; fi
if [ "$TGT" != "mh" -a "$TGT" != "sh" ]; then echo "unknown target language"; exit; fi
if [ "$SRC" == "$TGT" ]; then echo "source and target cannot be identical"; exit; fi
if [ "$SRC" \> "$TGT" ]; then echo "please ensure SRC < TGT"; exit; fi
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi


#
# Initialize tools and data paths
#

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
DATA_PATH=$PWD/data/umt/$DATA_FOLDER
PROC_PATH=$DATA_PATH/processed

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $PROC_PATH

# fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

# install tools
./install-tools.sh

# reload BPE codes
echo "Reloading BPE codes from $RELOAD_CODES ..."
cp $RELOAD_CODES $BPE_CODES

# reload full vocabulary
echo "Reloading vocabulary from $RELOAD_VOCAB ..."
cp $RELOAD_VOCAB $FULL_VOCAB

# preprocess
for lg in $SRC $TGT; do
  for split in "train" "valid" "test"; do
    RAW=$DATA_PATH/$split.$lg
    TOK=$RAW.tok
    BPE=$PROC_PATH/$split.$lg
    echo "Preprocessing $split.$lg..."
    if ! [[ -f "$TOK" ]]; then
      echo "Tokenizing $RAW..."
      eval "cat $RAW | $TOKENIZE en | python $LOWER_REMOVE_ACCENT > $TOK"
    fi
    if ! [[ -f "$BPE" ]]; then
      echo "Applying BPE codes to $split.$lg..."
      $FASTBPE applybpe $BPE $TOK $BPE_CODES $FULL_VOCAB
    fi
    if ! [[ -f "$BPE.pth" ]]; then
      echo "Binarizing $split.$lg..."
      $MAIN_PATH/preprocess.py $FULL_VOCAB $BPE
    fi
  done
done

#
# Link parallel validation and test data to monolingual data
#
for split in "train" "valid" "test"; do
    ln -sf $PROC_PATH/$split.$SRC.pth $PROC_PATH/$split.$SRC-$TGT.$SRC.pth
    ln -sf $PROC_PATH/$split.$TGT.pth $PROC_PATH/$split.$SRC-$TGT.$TGT.pth
done

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Parallel training data:"
echo "    $SRC: $PROC_PATH/train.$SRC.pth"
echo "    $TGT: $PROC_PATH/train.$TGT.pth"
echo "Parallel validation data:"
echo "    $SRC: $PROC_PATH/valid.$SRC.pth"
echo "    $TGT: $PROC_PATH/valid.$TGT.pth"
echo "Parallel test data:"
echo "    $SRC: $PROC_PATH/test.$SRC.pth"
echo "    $TGT: $PROC_PATH/test.$TGT.pth"
echo ""
