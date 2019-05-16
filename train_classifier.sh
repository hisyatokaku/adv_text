source variables.sh

if [ ! -d ${TRAIN_DIR} ]
then
    echo "making directory..."
    mkdir -p ${TRAIN_DIR}
fi

# bidir_lstm=Trueでpretrainしてもclassifierでbidir_lstm=Trueすると次元があわなくなる。

CUDA_VISIBLE_DEVICES=${GPU} python train_classifier.py \
       --train_dir=${TRAIN_DIR} \
       --pretrained_model_dir=${PRETRAIN_DIR} \
       --data_dir=${DATA_DIR} \
       --vocab_size=${VOCAB_SIZE}\
       --embedding_dims=256 \
       --rnn_cell_size=1024 \
       --bidir_lstm=False \
       --cl_num_layers=1 \
       --cl_hidden_size=30 \
       --batch_size=64 \
       --learning_rate=0.0005 \
       --learning_rate_decay_factor=0.9998 \
       --max_steps=${MAX_STEPS} \
       --max_grad_norm=1.0 \
       --num_timesteps=${NUM_TIMESTEPS} \
       --num_classes=${NUM_CLASSES} \
       --keep_prob_emb=0.5 \
       --normalize_embeddings=True \
       --adv_training_method='' \
       --perturb_norm_length=5.0

echo "making metadata..."
python dump_metadata.py \
       --TRAIN_DIR=${TRAIN_DIR} \
       --DATA_DIR=${DATA_DIR}
