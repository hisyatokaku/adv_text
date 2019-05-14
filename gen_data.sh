#IMDB_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB"
SST5_RAW_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/SST_data_extraction-master"
SST5_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed"

python gen_data.py \
       --output_dir="${SST5_DATA_DIR}" \
       --dataset=sst5 \
       --sst5_input_dir="${SST5_RAW_DIR}" \
       --lowercase=False \
       --label_gain=False
