import argparse
import os , sys
import random
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, Dataset, Subset, BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

#from dschat.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from gail_data_utils import CQADatasetLoader, SVAMPDatasetLoader, ASDivDatasetLoader,ESNLIDatasetLoader, get_shuffle_idx, MiniDataset, DataCollatorGAIL, PromptDataset
from gail_utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer, ExponentialMovingAverage
#from dschat.utils.module.lora import convert_lora_to_linear_layer
#from dschat.utils.perf import print_throughtput_step3
from deepspeed.accelerator import get_accelerator
from gail_engine import DeepSpeedGAILEngine
from gail_ppo_trainer import DeepSpeedGailPPOTrainer
#from gail_kd_evaluate import gail_evaluate


writer = None

def parse_args():
    global writer
    parser = argparse.ArgumentParser(description="GAIL-KD training arguments")
    parser.add_argument('--dataset_path', type=str, default="./dataset/cqa/")
    parser.add_argument('--data_split', type=str, default='2,4,4')
    parser.add_argument('--data_output_path', type=str)
    parser.add_argument('--unsupervised_dataset_name', type=str, default=None)
    parser.add_argument('--unsupervised_dataset_config_name', type=str, default=None)
    parser.add_argument('--discriminator_model_name_or_path', type=str, required=True)
    parser.add_argument('--generator_model_name_or_path', type=str, required=True)
    parser.add_argument('--critic_model_name_or_path', type=str, required=True)
    parser.add_argument('--num_padding_at_beginning', type=int, default=1)
    parser.add_argument('--per_device_generation_batch_size', type=int, default=16)
    parser.add_argument('--per_device_training_batch_size', type=int, default=16)
    parser.add_argument('--generation_batches', type=int, default=1)
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--max_answer_seq_len', type=int, default=256)
    parser.add_argument('--max_prompt_seq_len', type=int, default=256)
    parser.add_argument('--discriminator_learning_rate', type=float, default=5e-5)
    parser.add_argument('--generator_learning_rate', type=float, default=9.65e-6)
    parser.add_argument('--critic_learning_rate', type=float, default=5e-6)
    parser.add_argument('--discriminator_weight_decay', type=float, default=0.)
    parser.add_argument('--generator_weight_decay', type=float, default=0.)
    parser.add_argument('--critic_weight_decay', type=float, default=0.)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    # DeepSpeed
    parser.add_argument("--enable_hybrid_engine", action='store_true')
    parser.add_argument("--unpin_generator_parameters", action="store_true")
    parser.add_argument("--release_inference_cache", action="store_true")
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--tp_gather_partition_size", type=int, default=8)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--offload_reference_model", action="store_true")
    parser.add_argument("--generator_zero_stage", type=int, default=0)
    parser.add_argument("--critic_zero_stage", type=int, default=0)
    parser.add_argument("--generator_gradient_checkpointing", action="store_true")
    parser.add_argument("--critic_gradient_checkpointing", action="store_true")
    parser.add_argument("--discriminator_gradient_checkpointing", action="store_true")
    parser.add_argument("--generator_dropout", type=float, default=None)
    parser.add_argument("--discriminator_dropout", type=float, default=None)
    parser.add_argument("--critic_dropout", type=float, default=None)
    # LoRA
    parser.add_argument("--discriminator_lora_dim", type=int, default=0)
    parser.add_argument("--discriminator_lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument("--generator_lora_dim", type=int, default=0)
    parser.add_argument("--generator_lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument("--critic_lora_dim", type=int, default=0)
    parser.add_argument("--critic_lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument("--only_optimize_lora", action="store_true")
    parser.add_argument("--discriminator_lora_learning_rate", type=float, default=5e-4)
    parser.add_argument("--generator_lora_learning_rate", type=float, default=5e-4)
    parser.add_argument("--critic_lora_learning_rate", type=float, default=5e-4)
    # EMA
    parser.add_argument("--enable_ema", action="store_true")
    # Mixed Precision ZeRO++
    parser.add_argument("--enable_mixed_precision_lora", action="store_true")
    parser.add_argument("--compute_fp32_loss", action="store_true")
    # Tensorboard logging
    parser.add_argument("--enable_tensorboard", action="store_true")
    parser.add_argument("--tensorboard_path", type=str, default="gail_tensorboard")
    # Tokenizer
    parser.add_argument("--add_eot_token", action="store_true")
    # Generator/critic model overflow alignment
    parser.add_argument("--align_overflow", action="store_true")
    # print generator model answers during training
    parser.add_argument("--print_answers", action="store_true")
    parser.add_argument("--print_answers_interval", type=int, default=1)
    ## Testing
    parser.add_argument("--enable_test_mode", action="store_true")
    parser.add_argument("--test_stop_step", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.enable_tensorboard:
        print(f"Tensorboard logs going to: {args.tensorboard_path}/gail_tensorboard_logs")
        writer = SummaryWriter(f"{args.tensorboard_path}/gail_tensorboard_logs")
    # Validate settings
    if args.inference_tp_size > 1:
        assert(args.generator_zero_stage == 3), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"
    if args.generator_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.generator_lora_dim == 0:
        raise ValueError("The combination of [generator_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!")
    return args

def create_datasets(args, tokenizer):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    dataset_loader = None
    if "CQA" in args.dataset_path or "cqa" in args.dataset_path:
        dataset_loader = CQADatasetLoader()
    elif "SVAMP" in args.dataset_path or "svamp" in args.dataset_path:
        dataset_loader = SVAMPDatasetLoader()
    elif "anli1" in args.dataset_path or "asdiv" in args.dataset_path:
        dataset_loader = ASDivDatasetLoader()
    elif "ESNLI" in args.dataset_path or "esnli" in args.dataset_path:
        dataset_loader = ESNLIDatasetLoader()
    datasets = dataset_loader.load_from_json()
    if 'nli' in args.dataset_path:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )
    train_llms_output, train_labels = dataset_loader.load_llm_preds("train")
    test_llms_output, test_labels = dataset_loader.load_llm_preds("test")
    #prompt_train_dataset, _ = create_prompt_dataset(data)
    print(datasets["train"][0])
    print(datasets["train"][1])
    print(datasets["train"][2])
    print(len(datasets["train"]))
    print(train_llms_output[0])
    print(train_llms_output[1])
    print(train_llms_output[2])
    print(len(train_llms_output))
    #print("train_labels: ", train_labels)
    train_lens, test_lens = len(datasets["train"]), len(datasets["test"])
    train_datasets_len,test_datasets_len=len(datasets["train"]),len(datasets["test"])
    shuffled_train_index = get_shuffle_idx(args.seed, train_datasets_len)[:30]
    print(shuffled_train_index[:30])
    #train_dataset = Subset(datasets["train"], shuffled_train_index)
    #train_llm_output = Subset(train_llms_output, shuffled_train_index)
    #train_dataset = create_dataset_split()
    print(datasets["train"][0])
    print(train_llms_output[0])
    train_datasets, test_datasets = datasets["train"] , datasets["test"]
    train_tokens , train_llm_tokens = [] , []
    test_tokens , test_llm_tokens = [] , []
    for i in range(min(train_lens, 10000)):
        s0 = tokenizer(train_datasets[i]['input'], return_tensors="pt" )
        ys = tokenizer(train_llms_output[i], return_tensors="pt" )
        for key_word in ["input_ids", "attention_mask"]:
            s0[key_word] = s0[key_word].squeeze(0)
            ys[key_word] = ys[key_word].squeeze(0)
        train_tokens.append(s0)
        train_llm_tokens.append(ys)
    for i in range(min(test_lens, 1000)):
        t0 = tokenizer(test_datasets[i]['input'], return_tensors="pt")
        y0 = tokenizer(test_llms_output[i], return_tensors='pt')
        for key_word in ["input_ids", "attention_mask"]:
            t0[key_word] = t0[key_word].squeeze(0)
            y0[key_word] = y0[key_word].squeeze(0)
        test_tokens.append(t0)
        test_llm_tokens.append(y0)

    print("tokens")
    print(len(train_tokens))
    print(len(train_llm_tokens))
    print(train_tokens[0])
    print(train_tokens[1])
    print(train_llm_tokens[0])
    print(train_llm_tokens[1])
    print("test tokens ... ")
    print("len of test: ", len(test_tokens))
    print("len of test llm tokens: ", len(test_llm_tokens))
    print("test tokens 0: ", test_tokens[0])
    print("test tokens 1: ", test_tokens[1])
    print("test llm tokens 0: ", test_llm_tokens[0])
    print("test llm tokens 1: ", test_llm_tokens[1])
    print("pad token id: ", tokenizer.pad_token_id)
    print("eos : ", )
    #sys.exit(0)
    #train_tokens = PromptDataset(Subset(train_tokens, np.arange(len(train_tokens)).tolist()), tokenizer.pad_token_id)
    train_tokens = PromptDataset(train_tokens, tokenizer.pad_token_id)
    #train_llm_tokens = PromptDataset(Subset(train_llm_tokens, np.arange(len(train_llm_tokens)).tolist()), tokenizer.pad_token_id)
    train_llm_tokens = PromptDataset(train_llm_tokens, tokenizer.pad_token_id)
    #temp = Subset(train_llm_tokens, np.arange(len(train_llm_tokens)).tolist())
    #print("temp.indices")
    #print(temp.indices)
    print("train_tokens len: {}", len(train_tokens) )
    print("train_llm_tokens len: {}", len(train_llm_tokens) )
    #torch.save(train_tokens, "./output/temp1.pt")
    #train_tokens = torch.load("./output/temp1.pt")
    data_collator = DataCollatorGAIL(args.max_prompt_seq_len, args.inference_tp_size)
    '''
    if args.local_rank == -1:
        print("local_rank: {}", args.local_rank)
        #s0_train_sampler = BatchSampler( RandomSampler(train_tokens), batch_size=4, drop_last=True)
        s0_train_sampler = RandomSampler(train_tokens)
        #ys_train_sampler = BatchSampler( RandomSampler(train_llm_tokens), batch_size=4, drop_last=True)
        ys_train_sampler = RandomSampler(train_llm_tokens)
    else:
        s0_train_sampler = DistributedSampler(train_tokens)
        ys_train_sampler = DistributedSampler(train_llm_tokens)
    '''
    s0_train_sampler = BatchSampler(SequentialSampler(train_tokens), batch_size=1, drop_last=True)  #RandomSampler(train_tokens)
    ys_train_sampler = BatchSampler(SequentialSampler(train_llm_tokens), batch_size=1, drop_last=True) #RandomSampler(train_llm_tokens)
    train_dataloader = DataLoader(train_tokens,
                                collate_fn=data_collator,
                                #batch_sampler=s0_train_sampler,
                                sampler=s0_train_sampler,
                                #shuffle=False,
                                batch_size=args.per_device_generation_batch_size,
                                num_workers=0)
    train_llm_dataloader = DataLoader(train_llm_tokens,
                                collate_fn=data_collator,
                                #batch_sampler=ys_train_sampler,
                                sampler=ys_train_sampler,
                                #shuffle=False,
                                batch_size=args.per_device_generation_batch_size,
                                num_workers=0)
    num_update_steps_per_epoch = train_lens * (args.per_device_generation_batch_size / args.per_device_training_batch_size) * args.ppo_epochs / args.gradient_accumulation_steps 
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    s0_test_sampler = BatchSampler(SequentialSampler(test_tokens), batch_size=1, drop_last=True) #RandomSampler(test_tokens)
    ys_test_sampler = BatchSampler(SequentialSampler(test_llm_tokens), batch_size=1, drop_last=True) #RandomSampler(test_llm_tokens)
    test_dataloader = DataLoader(test_tokens,
                                collate_fn=data_collator,
                                #batch_sampler=s0_test_sampler,
                                sampler=s0_test_sampler,
                                #shuffle=False,
                                batch_size=args.per_device_generation_batch_size,
                                num_workers=0)
    test_llm_dataloader = DataLoader(test_llm_tokens,
                                collate_fn=data_collator,
                                #batch_sampler=ys_test_sampler,
                                sampler=ys_test_sampler,
                                #shuffle=False,
                                batch_size=args.per_device_generation_batch_size,
                                num_workers=0)


    return train_dataloader, train_llm_dataloader, num_total_iters, test_dataloader, test_llm_dataloader

def main():
    print("starting ... ")
    args = parse_args()
    device = torch.device(get_accelerator().device_name())
    set_random_seed(args.seed)
    #torch.distributed.barrier()
    #
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    print("get tokenizer ... ")
    tokenizer = load_hf_tokenizer(args.generator_model_name_or_path, fast_tokenizer=True, add_special_tokens=additional_special_tokens)
    #prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3)
    train_dataloader, train_llm_dataloader, num_total_iters, test_dataloader, test_llm_dataloader = create_datasets(args=args, tokenizer=tokenizer)
    gail_engine = DeepSpeedGAILEngine(args.generator_model_name_or_path, args.critic_model_name_or_path, args.discriminator_model_name_or_path, tokenizer, args, num_total_iters, device)
    # trainer ... 
    gail_trainer = DeepSpeedGailPPOTrainer(gail_engine, args)
    generation_mini_dataset = MiniDataset(args.generation_batches, args.per_device_training_batch_size)

    non_overflow_step_count = 0
    step_average_reward = 0
    ema_reward_score = ExponentialMovingAverage()

    for epoch in range(args.num_train_epochs):
        print(f"Beginning of Epoch {epoch}/{args.num_train_epochs}")
        for step , (batch_s0, batch_ys) in enumerate(zip(train_dataloader, train_llm_dataloader)):
            batch_s0 = to_device(batch_s0, device) # prompt
            batch_ys = to_device(batch_ys, device) # expert data
            #print("batch_s0")       
            #print(batch_s0)
            #print("batch_ys")
            #print(batch_ys)
            #sys.exit()
            #out, discrim_rewards, critic_values 
            #print("batch_s0 tokens 0")
            #print(batch_s0['tokens'][0])
            #print("batch_s0 token_att_mask 0")
            #print(batch_s0['token_att_mask'][0].sum())
            res = gail_trainer.generate_experience(batch_s0["tokens"], batch_s0["token_att_mask"], step)
            #print("res: ", res)
            #print("prompts: ", tokenizer.decode(res['prompts'][0], skip_special_tokens=True))
            print("output: ", tokenizer.decode(res['input_ids'][0], skip_special_tokens=True))
            #sys.exit()
            training_start = time.time()
            generation_dataset = generation_mini_dataset.add(res)
            #sys.exit(0)
            if generation_dataset is not None:
                inner_iter = 0
                generator_loss_sum, discriminator_loss_sum = 0 , 0
                critic_loss_sum = 0
                average_reward = 0

                ''' firstly , train discriminator, with a adam optimizer 
                '''
                #gail_trainer.discriminator_model() 
                #print("generation_dataset")
                #print(generation_dataset)
                loss_exp, loss_gen, loss_dis = gail_trainer.train_discriminator(batch_ys, generation_dataset[0], device)
                discriminator_loss_sum += loss_dis.item()
                temp = 0 

                for ppo_step in range(args.ppo_epochs):
                    ''' secondly , train generator and critic model,
                        with a ppo_step optimizer
                    '''
                    for i, generation_data in enumerate(zip(generation_dataset)):
                        ''' train gail generator and critic model in this funciton 
                            with ppo 
                        '''
                        #print(generation_data)
                        generator_loss, critic_loss = gail_trainer.train_gail(generation_data[0])
                        generator_loss_sum += generator_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += generation_data[0]["rewards"].mean()
                        
                        inner_iter += 1
                    
                    random.shuffle(generation_dataset)
                    #random.shuffle()
                end = time.time()
                training_time = end - training_start
                print("step: ", step)
                print("training_time: ", training_time)
                print("loss_exp: ", loss_exp)
                print("loss_gen: ", loss_gen)
                print("loss_dis: ", loss_dis)
                print("discriminator_loss_sum: ", discriminator_loss_sum)
                print("generator_loss: ", generator_loss_sum)
                print("critic_loss: ", critic_loss_sum)
                print("average_reward: ", average_reward)
    if args.output_dir is not None:
        print("save model ... ")
        save_hf_format(gail_engine.generator_model,
                       tokenizer,
                       args,
                       sub_folder="generator_model")
        save_hf_format(gail_engine.critic_model,
                       tokenizer,
                       args,
                       sub_folder="critic_model")
        save_hf_format(gail_engine.discriminator_model,
                       tokenizer,
                       args,
                       sub_folder="discriminator_model")
    '''
    'go to evaluate / test , below
    gail_evaluate(args, gail_trainer, test_dataloader, test_llm_dataloader, device)
    '''

if __name__ == "__main__":
    
    main()



