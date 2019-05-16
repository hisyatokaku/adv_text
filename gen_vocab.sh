source variables.sh

python gen_vocab.py \
       --output_dir=${DATA_DIR} \
       --dataset=${DATA} \
       --${DATA}_input_dir=${RAW_DIR} \
       --use_unlabeled=True \
       --lowercase=True