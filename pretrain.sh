TRAIN_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/train/sst5"
PRETRAIN_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/pretrain/sst5"
SST5_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed"

if [ ! -d PRETRAIN_SST5_DIR ]
then
    echo "making directory..."
    mkdir -p PRETRAIN_SST5_DIR
fi

CUDA_VISIBLE_DEVICES="1" python pretrain.py \
       --train_dir=$PRETRAIN_SST5_DIR \
       --data_dir=$SST5_DATA_DIR \
       --vocab_size=8752 \
       --bidir_lstm=True \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --num_candidate_samples=1024 \
       --batch_size=256 \
       --learning_rate=0.001 \
       --learning_rate_decay_factor=0.9999 \
       --max_steps=100000 \
       --max_grad_norm=1.0 \
       --num_timesteps=400 \
       --keep_prob_emb=0.5 \
       --num_classes=5 \
       --normalize_embeddings
