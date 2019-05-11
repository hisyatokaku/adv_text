PRETRAIN_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB/pretrain"
IMDB_DATA_DIR="/data/ishimochi0/tonkou/1_Dataset/IMDB"
CUDA_VISIBLE_DEVICES="1" python pretrain.py \
       --train_dir=$PRETRAIN_DIR \
       --data_dir=$IMDB_DATA_DIR \
       --vocab_size=87007 \
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
       --normalize_embeddings
