IMDB_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB"
SST5_RAW_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/SST_data_extraction-master"
SST5_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed"

python gen_vocab.py \
       --output_dir=$SST5_DATA_DIR \
       --dataset=sst5 \
       --sst5_input_dir="${SST5_RAW_DIR}" \
       --use_unlabeled=True \
       --lowercase=False
