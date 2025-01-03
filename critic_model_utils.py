import argparse
import os
import random
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from discrim_model import DiscriminatorModel
from gail_utils import load_state_dict_into_model, print_rank_0


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def create_hf_model(model_class, model_name_or_path, tokenizer, gail_training=False, dropout=None,device=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path), config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
    print("model.config.end_token_id: ", model.config.end_token_id)
    print("model.config.pad_token_id: ", model.config.pad_token_id)
    if device is not None:
        model = model.to(device)
    return model

def create_critic_model(model_name_or_path, tokenizer, num_padding_at_beginning=0, gail_training=False, dropout=None, compute_fp32_loss=False, prompt_length=128,device=None):
    import time
    start = time.time()
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer, gail_training, dropout)
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds", None)

    critic_model = DiscriminatorModel(critic_model, tokenizer, num_padding_at_beginning = num_padding_at_beginning, compute_fp32_loss = compute_fp32_loss, prompt_length = prompt_length)
    if device is not None:
        critic_model = critic_model.to(device)
    if gail_training:
        # load critic model from checkpoint
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(model_ckpt_path), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        print_rank_0(f">Creating model from_config took {end - start} seconds", None)
        
        # load critic model from checkpoint with zero-stage 3 compatibility
        start = time.time()
        load_state_dict_into_model(critic_model, model_ckpt_state_dict, "")
        end = time.time()

        print_rank_0(f">Creating model from_config took {end - start} seconds", None)
    
    return critic_model



