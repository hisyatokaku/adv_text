
source variables.sh

if [ ! -d ${EVAL_DIR} ]
then
    echo "making directory..."
    mkdir -p ${EVAL_DIR}
fi

CUDA_VISIBLE_DEVICES=${GPU} python evaluate.py \
       --eval_dir=${EVAL_DIR} \
       --checkpoint_dir=${TRAIN_DIR} \
       --eval_data=test \
       --run_once \
       --num_examples=25000 \
       --data_dir=${DATA_DIR} \
       --vocab_size=${VOCAB_SIZE} \
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --batch_size=256 \
       --num_timesteps=${NUM_TIMESTEPS} \
       --num_classes=${NUM_CLASSES} \
       --normalize_embeddings
