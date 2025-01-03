import argparse
import os , sys
import random
import time
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, RandomSampler, Dataset, Subset, BatchSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
from transformers import pipeline

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed 

from gail_data_utils import CQADatasetLoader, SVAMPDatasetLoader, ASDivDatasetLoader,ESNLIDatasetLoader, get_shuffle_idx, MiniDataset, DataCollatorGAIL, PromptDataset
from gail_utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer, ExponentialMovingAverage
from deepspeed.accelerator import get_accelerator
from gail_engine import DeepSpeedGAILEngineEvaluate
from gail_ppo_trainer import DeepSpeedGailEvaluate


def parse_args():
    parser = argparse.ArgumentParser(description="GAIL-KD training arguments")
    parser.add_argument('--dataset_path', type=str, default='./dataset/cqa')
    parser.add_argument('--model_name_or_path', type=str, default='./')
    parser.add_argument('--max_prompt_seq_len', type=int, default=128)
    parser.add_argument('--inference_tp_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args

def create_datasets(args, tokenizer):
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
    train_llms_output, train_labels = dataset_loader.load_llm_preds("train")
    test_llms_output, test_labels = dataset_loader.load_llm_preds("test")
    print(datasets["train"][0])
    print(datasets["train"][1])
    print(datasets["train"][2])
    print(len(datasets["train"]))
    print(train_llms_output[0])
    print(train_llms_output[1])
    print(train_llms_output[2])
    print(len(train_llms_output))
    train_lens, test_lens = len(datasets["train"]), len(datasets["test"])
    train_datasets_len,test_datasets_len=len(datasets["train"]),len(datasets["test"])
    print(datasets["train"][0])
    print(train_llms_output[0])
    train_datasets, test_datasets = datasets["train"] , datasets["test"]
    train_tokens, train_llm_tokens, test_tokens, test_llm_tokens=[],[],[],[]
    for i in range(train_lens):
        s0 = tokenizer(train_datasets[i]['input'], return_tensors="pt" )
        y0 = tokenizer(train_llms_output[i], return_tensors="pt" )
        for key_word in ["input_ids", "attention_mask"]:
            s0[key_word] = s0[key_word].squeeze(0)
            y0[key_word] = y0[key_word].squeeze(0)
        train_tokens.append(s0)
        train_llm_tokens.append(y0)
    for i in range(test_lens):
        t0 = tokenizer(test_datasets[i]['input'], return_tensors="pt")
        y0 = tokenizer(test_llms_output[i], return_tensors='pt')
        for key_word in ["input_ids", "attention_mask"]:
            t0[key_word] = t0[key_word].squeeze(0)
            y0[key_word] = y0[key_word].squeeze(0)
        test_tokens.append(t0)
        test_llm_tokens.append(y0)
    train_tokens = PromptDataset(train_tokens, tokenizer.pad_token_id)
    train_llm_tokens = PromptDataset(train_llm_tokens, tokenizer.pad_token_id)
    test_tokens = PromptDataset(test_tokens, tokenizer.pad_token_id)
    test_llm_tokens = PromptDataset(test_llm_tokens, tokenizer.pad_token_id)
    data_collator = DataCollatorGAIL(args.max_prompt_seq_len, args.inference_tp_size)
    s0_train_sampler=BatchSampler(SequentialSampler(train_tokens),batch_size=1,drop_last=True)
    y0_train_sampler=BatchSampler(SequentialSampler(train_llm_tokens),batch_size=1,drop_last=True)
    train_dataloader=DataLoader(train_tokens,collate_fn=data_collator,sampler=s0_train_sampler,batch_size=args.batch_size,num_workers=0)
    train_llm_dataloader=DataLoader(train_llm_tokens,collate_fn=data_collator,sampler=s0_train_sampler,batch_size=args.batch_size,num_workers=0)
    s0_test_sampler=BatchSampler(SequentialSampler(test_tokens),batch_size=1,drop_last=True)
    y0_test_sampler=BatchSampler(SequentialSampler(test_llm_tokens),batch_size=1,drop_last=True)
    test_dataloader=DataLoader(test_tokens,collate_fn=data_collator,sampler=s0_test_sampler,batch_size=args.batch_size,num_workers=0)
    test_llm_dataloader=DataLoader(test_llm_tokens,collate_fn=data_collator,sampler=y0_test_sampler,batch_size=args.batch_size,num_workers=0)
    return  train_datasets, train_llms_output, train_labels, test_datasets, test_llms_output, test_labels, train_dataloader, train_llm_dataloader, test_dataloader, test_llm_dataloader

def main():
    args = parse_args()
    end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = end_of_conversation_token 
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True, add_special_tokens=additional_special_tokens)
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    #configure_dropout(model_config, dropout)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=model_config)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    train_datasets, train_llms_output, train_labels, test_datasets, test_llms_output, \
        test_labels, train_tokens, train_llm_tokens, test_tokens, \
        test_llm_tokens = create_datasets(args, tokenizer)   
    
    prompts = train_datasets[0]['input'] + " " + train_llms_output[0] + " " \
            + train_datasets[1]['input'] + " " + train_llms_output[1] + " " \
            + train_datasets[2]['input'] + " " + train_llms_output[2] + " " \
            + train_datasets[3]['input'] + " " + train_llms_output[3] + "\n " 
    print("prompts: ", prompts)
    for i in range(len(test_datasets)):
        question = test_datasets[i]['input']
        inputs = tokenizer(prompts + question + '\n', return_tensors="pt", padding=True)
        #inputs = tokenizer(question , return_tensors="pt", padding=True)
        print("inputs: ", inputs)
        generate_ids = model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
                              max_new_tokens=100, do_sample=True, temperature=1.0)
        #print(generate_ids)
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        print("step: ", i, " question: ", question, " \nresult: ", result)

        if i >= 5: break

def main2():
    args = parse_args()
    device = torch.device(get_accelerator().device_name())
    end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = end_of_conversation_token 
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True, add_special_tokens=additional_special_tokens)
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    #configure_dropout(model_config, dropout)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=model_config).to(device)
    #model = AutoModelForCausalLM.from_config(model_config).to(device)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    train_datasets, train_llms_output, train_labels, test_datasets, test_llms_output, \
        test_labels, train_dataloader, train_llm_dataloader, test_dataloader, \
        test_llm_dataloader = create_datasets(args, tokenizer)   
    max_length=torch.tensor(128*2).to(device)
    #gail_engine = DeepSpeedGAILEngineEvaluate(args.model_name_or_path, tokenizer, args, num_total_iters=10)
    # trainer ... 
    #gail_trainer = DeepSpeedGailEvaluate(gail_engine, args)
    #generation_mini_dataset = MiniDataset(args.generation_batches, args.batch_size)


    for step, (batch_s0, batch_y0) in enumerate(zip(train_dataloader, train_llm_dataloader)) :
        batch_s0, batch_y0 = to_device(batch_s0, device), to_device(batch_y0, device)
        print("batch_s0 tokens: ", batch_s0['tokens'].shape)
        print("batch_s0 att: ", batch_s0['token_att_mask'].sum(-1))
        print(batch_s0['tokens'])
        print(batch_s0['token_att_mask'])
        res = model.generate(batch_s0['tokens'], attention_mask=batch_s0['token_att_mask'], max_length=max_length, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=1.0,no_repeat_ngram_size=2,)
        print("res: ", res.shape)
        print(res)
        #ids = tokenizer.batch_decode(res[:,128:], skip_special_tokens=True)
        ids = tokenizer.batch_decode(res, skip_special_tokens=True)
        #print("ids: ", ids[:, 128:])
        print("ids: ", ids)
        if step > 3:
            break

def main3():
    end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = end_of_conversation_token 
    device = "cuda"
    GENERATOR_MODEL_PATH="/home/yege/Work/compression/models/transformers/facebook/opt-350m"
    tokenizer = load_hf_tokenizer(GENERATOR_MODEL_PATH, fast_tokenizer=True, add_special_tokens=additional_special_tokens)
    #GENERATOR_MODEL_PATH="/home/yege/Work/compression/knowdis/gail-distill/output/generator_model_cqa_1"
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    generator = pipeline('text-generation', model=GENERATOR_MODEL_PATH,
                     device=local_rank)

    generator.model = deepspeed.init_inference(generator.model,
                                           tensor_parallel={"tp_size": world_size},
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)
    #generator.model = deepspeed.initialize(generator.model,)
    #prompts = "DeepSpeed is"
    prompts2 = "To keep track of your training progress, "
    prompts = "Mathematical logic is "
    print("prompts: ", prompts)
    batch = {}
    s0 = tokenizer(prompts, return_tensors="pt" ).to(device)
    s1 = tokenizer(prompts2, return_tensors="pt").to(device)
    
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn.functional as F
    pad_token_id = 2 #s0.input_ids[0]
    print("s0.input_ids")
    print(s0.input_ids)
    token_id = pad_sequence([s0.input_ids[0], s1.input_ids[0]], batch_first=True,padding_value=pad_token_id)
    token_mask = pad_sequence([s0.attention_mask[0], s1.attention_mask[0]], batch_first=True,padding_value=0)
    lengths = token_id.size()[-1]
    max_token_len = 10
    pad_length = max_token_len - lengths
    
    if pad_length > 0:
        batch['tokens'] = F.pad(token_id, pad=(0, pad_length),value=pad_token_id)
        batch["token_att_mask"] = F.pad(token_mask, pad=(0, pad_length),value=0)
    else:
        batch['tokens'] = token_id
        batch['token_att_mask'] = token_mask

    print(s0)
    string = generator.model.generate(batch["tokens"], attention_mask=batch["token_att_mask"], do_sample=True, min_length=75, max_length=100)
    print("output: ")
    print(string)
    #res = tokenizer.batch_decode(string[:, max_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    res = tokenizer.batch_decode(string, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("res: ", res)
    revert = tokenizer(res[0], return_tensors="pt").to(device)
    print("revert: ", revert)
    string3 = generator.model.generate(string,  do_sample=True, min_length=100,max_length=100)
    print("string3: ", string3)
    string2 = generator(prompts, do_sample=True, min_length=75,max_length=100)
    print("output 2: ", string2)


if __name__ == '__main__':
    
    #main2()
    main()


