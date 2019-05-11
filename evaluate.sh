EVAL_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB/eval"
TRAIN_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB/train"
IMDB_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB"
CUDA_VISIBLE_DEVICES="2" python evaluate.py \
       --eval_dir=$EVAL_DIR \
       --checkpoint_dir=$TRAIN_DIR \
       --eval_data=test \
       --run_once \
       --num_examples=25000 \
       --data_dir=$IMDB_DATA_DIR \
       --vocab_size=87007 \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --batch_size=256 \
       --num_timesteps=400 \
       --normalize_embeddings
