source variables.sh

CUDA_VISIBLE_DEVICES=${GPU} python pretrain.py \
       --train_dir=$PRETRAIN_DIR \
       --data_dir=$DATA_DIR \
       --vocab_size=${VOCAB_SIZE} \
       --bidir_lstm=False \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --num_candidate_samples=1024 \
       --batch_size=256 \
       --learning_rate=0.001 \
       --learning_rate_decay_factor=0.9999 \
       --max_steps=${MAX_STEPS} \
       --max_grad_norm=1.0 \
       --num_timesteps=${NUM_TIMESTEPS} \
       --keep_prob_emb=0.5 \
       --num_classes=${NUM_CLASSES} \
       --normalize_embeddings=True
