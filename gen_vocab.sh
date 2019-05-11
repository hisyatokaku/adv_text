IMDB_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB"
python gen_vocab.py \
       --output_dir=$IMDB_DATA_DIR \
       --dataset=imdb \
       --imdb_input_dir="${IMDB_DATA_DIR}/aclImdb" \
       --lowercase=False
