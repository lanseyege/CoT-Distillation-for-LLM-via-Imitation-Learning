#!/bin/bash
source /home/yege/anaconda3/bin/activate deepspeed
#Discriminator and Critic have the same model path

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yege/anaconda3/envs/deepspeed/lib/python3.8/site-packages/torch/lib/

GENERATOR_MODEL_PATH="/home/yege/Work/compression/models/transformers/facebook/opt-125m"
#GENERATOR_MODEL_PATH="/home/yege/Work/compression/knowdis/gail-distill-hf/output/generator_model_20240428-29"
#GENERATOR_MODEL_PATH="/home/yege/Work/compression/models/transformers/google/t5-v1_1-small"
CRITIC_MODEL_PATH="/home/yege/Work/compression/models/transformers/facebook/opt-125m"
#CRITIC_MODEL_PATH="/home/yege/Work/compression/models/transformers/google/t5-v1_1-small"

OUTPUT="./output"

Num_Padding_at_Beginning=1

Generator_Lr=9.65e-7
Discriminator_Lr=9.65e-7
Critic_Lr=5e-7

mkdir -p $OUTPUT
#--hostfile=hostfile --master_addr=192.168.63.134  --include="192.168.63.132:0@192.168.63.133:0@192.168.63.134:0" \

python main_gail_kd.py \
    --dataset_path ./datasets/esnli/ \
    --data_split 2,4,4 \
    --generator_model_name_or_path $GENERATOR_MODEL_PATH \
    --discriminator_model_name_or_path $CRITIC_MODEL_PATH \
    --critic_model_name_or_path $CRITIC_MODEL_PATH \
    --num_padding_at_beginning 1 \
    --per_device_generation_batch_size 4 \
    --per_device_training_batch_size 4 \
    --ppo_epochs 3 \
    --max_answer_seq_len 128 \
    --max_prompt_seq_len 128 \
    --generator_learning_rate ${Generator_Lr} \
    --discriminator_learning_rate ${Discriminator_Lr} \
    --critic_learning_rate ${Critic_Lr} \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --deepspeed --seed 123 \
    --generator_dropout 0.0 \
    --generator_lora_dim 128 \
    --discriminator_dropout 0.0 \
    --discriminator_lora_dim 128 \
    --enable_hybrid_engine \
    --local_rank -1 \
    --generator_lora_module_name decoder.layers. \
    --output_dir $OUTPUT \
    &> $OUTPUT/training.log

