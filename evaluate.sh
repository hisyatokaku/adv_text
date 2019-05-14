TRAIN_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/train/sst5_no_perb"
EVAL_SST5_DIR="/home/mil/tonkou/models/research/adversarial_text/eval/sst5"
SST5_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed"
CUDA_VISIBLE_DEVICES="0" python evaluate.py \
       --eval_dir=$EVAL_SST5_DIR \
       --checkpoint_dir=$TRAIN_SST5_DIR \
       --eval_data=test \
       --run_once \
       --num_examples=25000 \
       --data_dir=$SST5_DATA_DIR \
       --vocab_size=8752 \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --batch_size=256 \
       --num_timesteps=400 \
       --num_classes=5 \
       --normalize_embeddings
