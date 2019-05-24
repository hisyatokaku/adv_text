#!/usr/bin/env bash
DATA_DIR="/data/ishimochi0/tonkou/1_Dataset"
EXPERIMENT_DIR="/home/mil/tonkou/models/research/adversarial_text"

IMDB_RAW_DIR="${DATA_DIR}/IMDB/aclImdb"
IMDB_DATA_DIR="${DATA_DIR}/IMDB/adv_processed"
IMDB_PRETRAIN_DIR="${EXPERIMENT_DIR}/pretrain/Imdb/unidir_short"
IMDB_TRAIN_DIR="${EXPERIMENT_DIR}/train/Imdb/unidir_short"
IMDB_EVAL_DIR="${EXPERIMENT_DIR}/eval/Imdb/unidir_short"

SST5_RAW_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/SST_data_extraction-master"
SST5_DATA_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/adv_processed"
SST5_PRETRAIN_DIR="${EXPERIMENT_DIR}/pretrain/sst5/unidir"
SST5_TRAIN_DIR="${EXPERIMENT_DIR}/train/sst5/unidir_at_norm5"
SST5_EVAL_DIR="${EXPERIMENT_DIR}/eval/sst5/unidir_at_norm5"

SST2_RAW_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/SST_data_extraction-master"
SST2_DATA_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/adv_processed_sst2"
SST2_PRETRAIN_DIR="${EXPERIMENT_DIR}/pretrain/sst2/unidir"
SST2_TRAIN_DIR="${EXPERIMENT_DIR}/train/sst2/unidir_at_norm1_outputall"
SST2_EVAL_DIR="${EXPERIMENT_DIR}/eval/sst2/unidir_at_norm1_test"

SST2_PERB_TEST_DATA_DIR="${DATA_DIR}/4_sst/stanfordSentimentTreebank/perb_test_adv_processed_sst2"

DATA="sst2"
GPU="0"

if [ $DATA = "sst5" ]; then
    RAW_DIR=${SST5_RAW_DIR}
    DATA_DIR=${SST5_DATA_DIR}
    PRETRAIN_DIR=${SST5_PRETRAIN_DIR}
    TRAIN_DIR=${SST5_TRAIN_DIR}
    EVAL_DIR=${SST5_EVAL_DIR}
    MAX_STEPS="2000"
    NUM_CLASSES="5"
    NUM_TIMESTEPS="40"

elif [ $DATA = "sst2" ]; then
    RAW_DIR=${SST2_RAW_DIR}
    DATA_DIR=${SST2_DATA_DIR}
    PRETRAIN_DIR=${SST2_PRETRAIN_DIR}
    TRAIN_DIR=${SST2_TRAIN_DIR}
    EVAL_DIR=${SST2_EVAL_DIR}
    MAX_STEPS="2000"
    NUM_CLASSES="2"
    NUM_TIMESTEPS="40"

elif [ $DATA = "imdb" ]; then
    RAW_DIR=${IMDB_RAW_DIR}
    DATA_DIR=${IMDB_DATA_DIR}
    PRETRAIN_DIR=${IMDB_PRETRAIN_DIR}
    TRAIN_DIR=${IMDB_TRAIN_DIR}
    EVAL_DIR=${IMDB_EVAL_DIR}
    MAX_STEPS="100000"
    NUM_CLASSES="2"
    NUM_TIMESTEPS="40"
else
    echo "unknown dataset."
    exit 1
fi

VOCAB_PATH="${DATA_DIR}/vocab.txt"
VOCAB_SIZE="$(cat ${VOCAB_PATH} | wc -l )"
echo "vocab size: ${VOCAB_SIZE}"
