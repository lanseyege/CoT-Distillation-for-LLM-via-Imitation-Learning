
#MODEL_PATH="/home/yege/Work/compression/models/transformers/facebook/opt-1.3b"
MODEL_PATH="/home/yege/Work/compression/knowdis/gail-distill-hf/output/generator_model/"
#MODEL_PATH="/home/yege/Work/compression/models/transformers/facebook/opt-125m"
DATA_PATH="./dataset/cqa"

python main_evaluate.py --dataset_path ${DATA_PATH} --model_name_or_path ${MODEL_PATH}

