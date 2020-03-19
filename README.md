# UnsupervisedDecomposition

PyTorch original implementation of "[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)."

TL;DR: We decompose hard (multi-hop) questions into several, easier (single-hop) questions using unsupervised learning. Our decompositions improve multi-hop QA on [HotpotQA](https://arxiv.org/pdf/1809.09600.pdf) without requiring extra supervision to decompose questions.  

## Overview
`XLM` contains the code to train (Unsupervised) Seq2Seq models, based on the code from [XLM](https://github.com/facebookresearch/XLM).
We made the following changes/additions:
- Unsupervised stopping criterion
- Tensorboard logging
- Data preprocessing scripts
- Minor bug fixes from original XLM code
- When initializing a smaller Seq2Seq model with XLM_en pretrained weights, automatically initialize the encoder with the first XLM_en layer weights and the decoder with the remaining layer weights. 

`pytorch-transformers` contains the code to train question answering models (single-hop and multi-hop), based on the code from [transformers](https://github.com/huggingface/transformers). We made the following additions:
- Scripts/notebooks to preprocess data
- Additions to evaluation to handle/evaluate on HotpotQA (i.e., extend single-paragraph SQuAD implementation to multi-paragraph setting)

## Installation

Create an anaconda3 environment (we used anaconda3 version 5.0.1):
```bash
conda create -y -n UnsupervisedDecomposition python=3.7
conda activate UnsupervisedDecomposition
# Install PyTorch 1.0. We used CUDA 10.0 (with NCCL/2.4.7-1) (see https://pytorch.org/ to install with other CUDA versions):
conda install -y pytorch=1.0 torchvision cudatoolkit=10.0 -c pytorch
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA 10.0
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # Download Spacy model for NER
```

If your hardware supports half-precision (fp16), you can install NVIDIA [apex](https://github.com/NVIDIA/apex) to speed up QA model training.
Also, set the `MAIN_DIR` variable to point to the main directory for this repo, e.g.:
```bash
export MAIN_DIR=/path/to/UnsupervisedDecomposition
```

## Downloading and Preprocessing Data
Run `download_data.sh` once, to download/prepare the necessary files for decomposition and question answering training, e.g.:
```bash
bash download_data.sh --main_dir $MAIN_DIR
```

See below to train a decomposition model, or skip to "QA Model Training" to train a question answering model given our trained decomposition model (`XLM/dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/best-valid_mlm_ppl.pth`).
You can view our generations from the model in the downloaded files `XLM/dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/hyp.st=0.0.bs=5.lp=1.0.es=False.seed=0.mh-sh.{train|valid}.pred.bleu.sh.txt`. 

## Unsupervised Decomposition Training
Create pseudo-decomposition training data using FastText embeddings and entity replacement using `create_pseudo_decompositions.sh `, e.g.:
```bash
bash create_pseudo_decompositions.sh --main_dir $MAIN_DIR
```

Then, train an Unsupervised Seq2Seq model as follows (initializing from our pre-trained MLM model):
```bash
# Set the following parameters based on your hardware
export NPROC_PER_NODE=8  # Use 1 for single-GPU training
export N_NODES=1  # Use >1 for multi-node training (where each node has NPROC_PER_NODE GPUs)
BS=32  # Make batch size smaller if GPU goes out-of-memory. Effective batch size is BS*NPROC_PER_NODE*N_NODES

# Select an MLM initialization checkpoint (for now, let's load the MLM we already pre-trained)
MLM_INIT=dumped/mlm.dev1.pseudo_decomp_random.mined/best-valid_mlm_ppl.pth 

# Train USeq2Seq model
export NGPU=$NPROC_PER_NODE
if [[ $NPROC_PER_NODE -gt 1 ]]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE"; else DIST_OPTS=""; fi
NUM_TRAIN=`wc -l < data/umt/$DATA_FOLDER/processed/train.mh`
python $DIST_OPTS train.py --exp_name umt.$DATA_FOLDER --data_path data/umt/$DATA_FOLDER/processed --dump_path ./dumped/ --reload_model "$MLM_INIT,$MLM_INIT" --encoder_only false --emb_dim 2048 --n_layers 6 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --use_lang_emb true --lgs 'mh-sh' --ae_steps 'mh,sh' --bt_steps 'mh-sh-mh,sh-mh-sh' --stopping_criterion 'valid_mh-sh-mh_mt_effective_goods_back_bleu,2' --validation_metrics 'valid_mh-sh-mh_mt_effective_goods_back_bleu' --eval_bleu true --epoch_size $((4*NUM_TRAIN/(NPROC_PER_NODE*N_NODES))) --lambda_ae '0:1,100000:0.1,300000:0' --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00003' --tokens_per_batch 1024 --batch_size $BS --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --max_len 128 --bptt 128 --save_periodic 0 --split_data true --validation_weight 0.5
```

### Seq2Seq Decomposition Training (Optional)
Alternatively, you can train a standard Seq2Seq model as follows:
```bash
export NPROC_PER_NODE=8  # Use 1 for single-GPU training
export N_NODES=1  # Use >1 for multi-node training (where each node has NPROC_PER_NODE GPUs)
BS=128  # Make batch size smaller if GPU goes out-of-memory. Effective batch size is BS*NPROC_PER_NODE*N_NODES

MLM_INIT=dumped/mlm.dev1.pseudo_decomp_random.mined/best-valid_mlm_ppl.pth
export NGPU=$NPROC_PER_NODE
if [[ $NPROC_PER_NODE -gt 1 ]]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE"; else DIST_OPTS=""; fi
DATA_FOLDER=dev1.pseudo_decomp.replace_entity_by_type
DATA_PATH=data/umt/$DATA_FOLDER/processed
NUM_TRAIN=`wc -l < $DATA_PATH/train.mh`
mkdir -p $OUTPUT_DIR
python $DIST_OPTS train.py --exp_name mt.$DATA_FOLDER --data_path $DATA_PATH --dump_path ./dumped/ --reload_model "$MLM_INIT,$MLM_INIT" --encoder_only false --emb_dim 2048 --n_layers 6 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --use_lang_emb true --lgs 'mh-sh' --mt_steps 'mh-sh,sh-mh' --stopping_criterion 'valid_mh-sh_mt_bleu,2' --validation_metrics 'valid_mh-sh_mt_bleu' --eval_bleu true --epoch_size $((2*NUM_TRAIN/(NPROC_PER_NODE*N_NODES))) --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001' --tokens_per_batch 1024 --batch_size $BS --max_len 128 --bptt 128 --split_data true
```

You can also use the trained Seq2Seq model checkpoint as the pre-trained initialization (`MLM_INIT`) for USeq2Seq training, as our Curriculum Seq2Seq approach does (see Appendix).

### MLM Pre-training (Optional)
To pre-train your own MLM initialization (used as `MLM_INIT`), use the below commands:
```bash
# Set the following parameters based on your hardware
export NPROC_PER_NODE=8  # Use 1 for single-GPU training
export N_NODES=8  # Use >1 for multi-node training (where each node has NPROC_PER_NODE GPUs)

# Copy XLM's English pre-trained MLM weights, which we use to initialize our MLM training
wget https://dl.fbaipublicfiles.com/XLM/mlm_en_2048.pth
mv mlm_en_2048.pth dumped/xlm_en/

# MLM pre-training (on same data as above)
export NGPU=$NPROC_PER_NODE
if [[ $NPROC_PER_NODE -gt 1 ]]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE"; else DIST_OPTS=""; fi
EPOCH_SIZE=$((2*NUM_TRAIN))
BS=24
EFFECTIVE_BS=$((BS*NPROC_PER_NODE*N_NODES))
NUM_TRAIN=`wc -l < data/umt/$DATA_FOLDER/processed/train.mh`
# For fp16: Add "--fp16 true --amp 1" below
python $DIST_OPTS train.py --exp_name mlm.$DATA_FOLDER --data_path data/umt/$DATA_FOLDER/processed --dump_path ./dumped/ --reload_model 'dumped/xlm_en/mlm_en_2048.pth' --emb_dim 2048 --n_layers 12 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --use_lang_emb true --lgs 'mh-sh' --clm_steps '' --mlm_steps 'mh,sh' --stopping_criterion '_valid_mlm_ppl,0' --validation_metrics '_valid_mlm_ppl' --epoch_size $EPOCH_SIZE --optimizer "adam_inverse_sqrt,lr=0.00003,beta1=0.9,beta2=0.98,weight_decay=0,warmup_updates=$((EPOCH_SIZE/EFFECTIVE_BS))" --batch_size $BS --max_len 128 --bptt 128 --accumulate_gradients 1 --word_pred 0.15 --sample_alpha 0
```

## QA Model Training
With a trained decomposition model, we can generate decompositions for multi-hop questions (train and valid sets), and train a question answering model to use the decompositions (below we use our pre-trained decomposition model which you downloaded):
```bash
# Generate decompositions
ST=0.0
LP=1.0
BEAM=5
SEED=0
# Point to model directory (change the final directory number/string/id below to match the directory string from the previous Unsupervised Seq2Seq training command)
MODEL_DIR=dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223
MODEL_NO="$(echo $MODEL_DIR | rev | cut -d/ -f1 | rev)"
for SPLIT in valid train; do
    # Note: Decrease batch size below if GPU goes out of memory
    cat data/umt/all/processed/$SPLIT.mh | python translate.py --exp_name translate --src_lang mh --tgt_lang sh --model_path $MODEL_DIR/best-valid_mh-sh-mh_mt_effective_goods_back_bleu.pth --output_path $MODEL_DIR/$SPLIT.pred.bleu.sh --batch_size 48 --beam_size $BEAM --length_penalty $LP --sample_temperature $ST
done

# Convert Sub-Qs to SQUAD format
cd $MAIN_DIR/pytorch-transformers
for SPLIT in valid train; do
    python umt_gen_subqs_to_squad_format.py --model_dir $MODEL_DIR --data_folder all --sample_temperature $ST --beam $BEAM --length_penalty $LP --seed $SEED --split $SPLIT --new_data_format
done

# Answer sub-Qs
DATA_FOLDER=data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED
for SPLIT in "dev" "train"; do
for NUM_PARAGRAPHS in 1 3; do
    # For fp16: Add "--fp16 --fp16_opt_level O2" below
    python examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --train_file $DATA_FOLDER/train.json --predict_file $DATA_FOLDER/$SPLIT.json --do_eval --do_lower_case --version_2_with_negative --output_dir checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 32 --learning_rate 1.5e-5 --max_query_length 234 --max_seq_length 512 --doc_stride 50 --num_shards 1 --seed 0 --max_grad_norm inf --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs 2 --write_dir $DATA_FOLDER/roberta_predict.np=$NUM_PARAGRAPHS --no_answer_file
done
done

# Ensemble sub-answer predictions
for SPLIT in "dev" "train"; do
    python ensemble_answers_by_confidence_script.py --seeds_list 1 3 --no_answer_file --split $SPLIT --preds_file1 data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED/roberta_predict.np={}/nbest_predictions_$SPLIT.json
done

# Add sub-questions and sub-answers to QA input
FLAGS="--atype sentence-1-center --subq_model roberta-large-np=1-3 --use_q --use_suba --use_subq"
python add_umt_subqs_subas_to_q_squad_format_new.py --subqs_dir data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED --splits train dev --num_shards 1 --model_dir $MODEL_DIR --sample_temperature $ST --beam $BEAM --length_penalty $LP --seed $SEED --subsample_data --use_easy --use_squad $FLAGS

# Train QA model
export NGPU=8  # Set based on number of available GPUs
if [ $NGPU -gt 1 ]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NGPU"; else DIST_OPTS=""; fi
if [ $NGPU -gt 1 ]; then EVAL_OPTS="--do_eval"; else EVAL_OPTS=""; fi
export MASTER_PORT=$(shuf -i 12001-19999 -n 1)
FLAGS_STRING="${FLAGS// --/.}"
FLAGS_STRING="${FLAGS_STRING//--/.}"
FLAGS_STRING="${FLAGS_STRING// /=}"
TN=hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED$FLAGS_STRING.suba1=0.suba2=0-squad.medium_hard_frac=1.0
RANDOM_SEED=0
OUTPUT_DIR="checkpoint/tn=$TN/rs=$RANDOM_SEED"
# For fp16: Add "--fp16 --fp16_opt_level O2" below
python $DIST_OPTS examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --train_file data/$TN/train.json --predict_file data/$TN/dev.json --do_train $EVAL_OPTS --do_lower_case --version_2_with_negative --output_dir $OUTPUT_DIR --per_gpu_train_batch_size $((64/NGPU)) --per_gpu_eval_batch_size 32 --learning_rate 1.5e-5 --master_port $MASTER_PORT --max_query_length 234 --max_seq_length 512 --doc_stride 50 --num_shards 1 --seed $RANDOM_SEED --max_grad_norm inf --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs 2 --overwrite_output_dir
```

## Creating Alternate Pseudo-Decompositions
We can also create pseudo-decompositions using other embedding methods aside from FastText, as described in the Appendix.
To do so, use the functions in `pytorch-transformers/pseudoalignment/pseudo_decomp_{paired_random|fasttext|tfidf|bert|variable}.py`, e.g., by running:
```angular2html
python pseudoalignment/pseudo_decomp_fasttext.py \
    --split train    # decompose the hotpotQA training question
    --min_q_len 4    # minimum length of short questions (tokens)
    --max_q_len 20   # maximum length of short questions (tokens)
    --beam_size 100  # subset of short questions to search exhaustively over for each complex question
    --data_folder data/umt/decomposition_name  # path to dump the results to
```

The different pseudo-decomposition methods are:
* `pseudo_decomp_fasttext.py` - decompose using bag of fasttext vectors
* `pseudo_decomp_random.py` - randomly pair short questions (for ablations/comparisons)
* `pseudo_decomp_tfidf.py` - decompose using bag of tfidf vectors
* `pseudo_decomp_variable.py` - decompose using bag of facttext vectors, but using a variable number of subquestions (see Appendix)
* `pseudo_decomp_bert.py` -  decompose using bert embeddings (requires generating the bert embeddings first with `embed_questions_with_bert.py`)
* `pseudo_decomp_bert_nsp.py` -  decompose using bert NSP embeddings (not in the paper) (requires generating the bert embeddings first with `embed_questions_with_bert.py`)

## Variable Number of Sub-Questions
To train a decomposition model to generate a variable number of sub-questions, you'll need to make the following changes:
- Train on variable-length pseudo-decompositions, created using `python pseudoalignment/pseudo_decomp_fasttext.py` (see above).
- Use a version of the unsupervised stopping criterion which only counts bad decompositions as those with `N<2` sub-questions (as opposed to `N!=2` sub-questions). Simply add the flag `--one_to_variable` when training (Unsupervised) Seq2Seq models with `XLM/train.py`.
- Have the single-hop QA model answer an arbitrary number of sub-questions, instead of a maximum of 2 sub-questions. Simply add `--one_to_variable` to the `FLAGS` variable used in the "QA Model Training" section earlier.

## Citation
```
@article{perez2020unsupervised,
    title={Unsupervised Question Decomposition for Question Answering},
    author={Ethan Perez and Patrick Lewis and Wen-tau Yih and Kyunghyun Cho and Douwe Kiela},
    year={2020},
    eprint={2002.09758},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2002.09758}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
