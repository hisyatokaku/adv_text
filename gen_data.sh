source variables.sh

python gen_data.py \
       --output_dir="${DATA_DIR}" \
       --dataset=${DATA} \
       --${DATA}_input_dir="${RAW_DIR}" \
       --lowercase=True \
       --label_gain=False
