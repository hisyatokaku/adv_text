TRAIN_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/train/sst5_no_perb"
PRETRAIN_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/pretrain/sst5"
SST5_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed"

if [ ! -d TRAIN_SST5_DIR ]
then
    echo "making directory..."
    mkdir -p TRAIN_SST5_DIR
fi

CUDA_VISIBLE_DEVICES="1" python train_classifier.py \
       --train_dir=$TRAIN_SST5_DIR \
       --pretrained_model_dir=$PRETRAIN_SST5_DIR \
       --data_dir=$SST5_DATA_DIR \
       --vocab_size=8752 \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --bidir_lstm=True \
       --cl_num_layers=1 \
       --cl_hidden_size=30 \
       --batch_size=64 \
       --learning_rate=0.0005 \
       --learning_rate_decay_factor=0.9998 \
       --max_steps=15000 \
       --max_grad_norm=1.0 \
       --num_timesteps=40 \
       --num_classes=5 \
       --keep_prob_emb=0.5 \
       --normalize_embeddings \
       --adv_training_method='' \
       --perturb_norm_length=5.0
