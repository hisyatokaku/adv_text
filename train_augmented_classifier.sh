#!/usr/bin/env bash

DATA_DIR="/data/ishimochi0/tonkou/1_Dataset"
EXPERIMENT_DIR="/home/mil/tonkou/models/research/adversarial_text"

SST2_RAW_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/SST_data_extraction-master"
SST2_AUGMENTED_DATA_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/augmented_adv_processed_sst2"
SST2_PRETRAIN_DIR="${EXPERIMENT_DIR}/pretrain/sst2/unidir"
SST2_TRAIN_DIR="${EXPERIMENT_DIR}/train/sst2/unidir_at_norm1_augmented"
SST2_AUGMENTED_EVAL_DIR="${EXPERIMENT_DIR}/eval/sst2/unidir_at_norm1_augmented"

DATA="sst2"
GPU=""

MAX_STEPS="2000"
NUM_TIMESTEPS="40"
NUM_CLASSES="2"

VOCAB_PATH="${SST2_AUGMENTED_DATA_DIR}/vocab.txt"
VOCAB_SIZE="$(cat ${VOCAB_PATH} | wc -l )"
echo "vocab size: ${VOCAB_SIZE}"

python gen_vocab.py \
       --output_dir=${SST2_AUGMENTED_DATA_DIR} \
       --dataset=${DATA} \
       --${DATA}_input_dir=${SST2_RAW_DIR} \
       --use_unlabeled=False \
       --lowercase=True

python gen_data.py \
       --output_dir="${SST2_AUGMENTED_DATA_DIR}" \
       --dataset=${DATA} \
       --${DATA}_input_dir="${SST2_RAW_DIR}" \
       --lowercase=True \
       --label_gain=False

CUDA_VISIBLE_DEVICES=${GPU} python train_classifier.py \
       --train_dir=${SST2_TRAIN_DIR} \
       --pretrained_model_dir=${SST2_PRETRAIN_DIR} \
       --data_dir=${DATA_DIR} \
       --vocab_size=${VOCAB_SIZE}\
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --bidir_lstm=False \
       --cl_num_layers=1 \
       --cl_hidden_size=30 \
       --batch_size=64 \
       --learning_rate=0.0005 \
       --learning_rate_decay_factor=0.9998 \
       --max_steps=${MAX_STEPS} \
       --max_grad_norm=1.0 \
       --num_timesteps=${NUM_TIMESTEPS} \
       --num_classes=${NUM_CLASSES} \
       --keep_prob_emb=0.5 \
       --normalize_embeddings=True \
       --adv_training_method='at' \
       --perturb_norm_length=1.0

echo "making metadata..."
python dump_metadata.py \
       --TRAIN_DIR=${SST2_TRAIN_DIR} \
       --DATA_DIR=${SST2_AUGMENTED_DATA_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python evaluate.py \
       --eval_dir=${SST2_AUGMENTED_EVAL_DIR} \
       --checkpoint_dir=${SST2_TRAIN_DIR} \
       --eval_data=test \
       --run_once \
       --num_examples=25000 \
       --data_dir=${SST2_AUGMENTED_DATA_DIR} \
       --vocab_size=${VOCAB_SIZE} \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --batch_size=256 \
       --num_timesteps=${NUM_TIMESTEPS} \
       --num_classes=${NUM_CLASSES} \
       --normalize_embeddings


