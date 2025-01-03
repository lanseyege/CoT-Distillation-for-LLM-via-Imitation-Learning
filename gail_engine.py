import time
import torch
from transformers import AutoModelForCausalLM, get_scheduler

from torch.optim import AdamW

from gail_ds_utils import get_train_ds_config, get_eval_ds_config
from critic_model_utils import create_hf_model, create_critic_model
from gail_utils import get_optimizer_grouped_parameters


def log_init(model_name, stime = None):
    return time.time()

class DeepSpeedGAILEngine():

    def __init__(self, generator_model_name_or_path, critic_model_name_or_path, discriminator_model_name_or_path, tokenizer, args, num_total_iters, device):
        self.args = args
        self.device = device
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        self.generator_model, self.generator_optim, self.generator_lr_scheduler = self._init_generator(generator_model_name_or_path = generator_model_name_or_path)
        self.ref_generator_model = self._init_generator_ref(generator_model_name_or_path = generator_model_name_or_path)
        self.critic_model, self.critic_optim, self.critic_lr_scheduler = self._init_critic(critic_model_name_or_path = critic_model_name_or_path)
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()
        self.discriminator_model, self.discriminator_optim, self.discriminator_lr_scheduler = self._init_discriminator(discriminator_model_name_or_path = discriminator_model_name_or_path)
        if self.args.discriminator_gradient_checkpointing:
            self.discriminator.gradient_checkpointing_enable()

    def _init_generator(self, generator_model_name_or_path):
        stime = log_init("gail_generator")
        # Model 
        generator_model = create_hf_model(
            model_class = AutoModelForCausalLM,
            model_name_or_path = generator_model_name_or_path,
            tokenizer = self.tokenizer,
            gail_training = True,
            dropout = self.args.generator_dropout,
            device = self.device)
        # LoRA
        '''
        if self.args.generator_lora_dim > 0 : 
            generator_model = convert_linear_layer_to_lora(
                generator_model, self.args.generator_lora_module_name, self.args.generator_lora_dim)
            if self.args.only_optimize_lora:
                generator_model = only_optimize_lora_parameters(generator_model)
                generator_model = make_model_gradient_checkpointing_compatible(generator_model)
        '''
        # Optimizer 
        optim_params = get_optimizer_grouped_parameters(generator_model, self.args.generator_weight_decay, self.args.generator_lora_learning_rate)
        optim = AdamW(optim_params, 
                            lr = self.args.generator_learning_rate,
                            betas = (0.9, 0.95))
        # LR Scheduler
        lr_scheduler = get_scheduler(
                name = self.args.lr_scheduler_type, 
                optimizer = optim, 
                num_warmup_steps = self.args.num_warmup_steps,
                num_training_steps = self.num_total_iters)

        return generator_model, optim, lr_scheduler

    def _init_generator_ref(self, generator_model_name_or_path):
        stime = log_init("Ref")
        ref_model = create_hf_model(AutoModelForCausalLM,
                                    generator_model_name_or_path,
                                    self.tokenizer,
                                    device=self.device)
        return ref_model

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("gail_critic")
        # Model 
        critic_model = create_critic_model(
                model_name_or_path = critic_model_name_or_path,
                tokenizer = self.tokenizer,
                num_padding_at_beginning = self.args.num_padding_at_beginning,
                gail_training = True,
                dropout = self.args.critic_dropout,
                compute_fp32_loss=False,
                prompt_length = self.args.max_prompt_seq_len,
                device=self.device)
        # LoRA
        '''
        if self.args.critic_lora_dim > 0 : 
            critic_model = convert_linear_layer_to_lora(
                    critic_model, self.args.critic_lora_module_name,
                    self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)
                critic_model = make_model_gradient_checkpointing_compatible(critic_model)
        '''
        # Optimizer
        #AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
                critic_model, self.args.critic_weight_decay,
                self.args.critic_lora_learning_rate)
        optim = AdamW(optim_params, lr = self.args.critic_learning_rate, betas = (0.9, 0.95))
        # LR Scheduler
        lr_scheduler = get_scheduler(
                name = self.args.lr_scheduler_type,
                optimizer = optim,
                num_warmup_steps = self.args.num_warmup_steps,
                num_training_steps = self.num_total_iters,)
        # DeepSpeed Engine
        stime = log_init("gail_critic", stime = stime)
        return critic_model, optim, lr_scheduler
        
    def _init_discriminator(self, discriminator_model_name_or_path):
        stime = log_init("gail_discriminator")
        # Model
        discriminator_model = create_critic_model(
                model_name_or_path = discriminator_model_name_or_path,
                tokenizer = self.tokenizer,
                num_padding_at_beginning = self.args.num_padding_at_beginning,
                gail_training = True,
                dropout = self.args.discriminator_dropout,
                compute_fp32_loss=False,
                prompt_length=self.args.max_prompt_seq_len,
                device=self.device)
        # LoRA
        '''
        if self.args.discriminator_lora_dim > 0 : 
            discriminator_model = convert_linear_layer_to_lora(
                    discriminator_model, self.args.discriminator_lora_module_name,
                    self.args.discriminator_lora_dim)
            if self.args.only_optimize_lora:
                discriminator_model = only_optimize_lora_parameters(critic_model)
                discriminator_model = make_model_gradient_checkpointing_compatible(discriminator_model)
        '''
        # Optimizer
        #AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
                discriminator_model, self.args.discriminator_weight_decay,
                self.args.discriminator_lora_learning_rate)
        optim = AdamW(optim_params, lr = self.args.discriminator_learning_rate, betas = (0.9, 0.95))
        # LR Scheduler
        lr_scheduler = get_scheduler(
                name = self.args.lr_scheduler_type,
                optimizer = optim,
                num_warmup_steps = self.args.num_warmup_steps,
                num_training_steps = self.num_total_iters,)
        log_init("Discriminator", stime = stime)
        return discriminator_model, optim, lr_scheduler

class DeepSpeedGAILEngineEvaluate():

    def __init__(self, generator_model_name_or_path, tokenizer, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        self.generator , self.generator_model = self._init_generator(generator_model_name_or_path = generator_model_name_or_path)

    def _init_generator(self, generator_model_name_or_path):
        stime = log_init("gail_generator")
        ds_config = get_train_ds_config(
                offload = self.args.offload,
                dtype = self.args.dtype,
                stage = self.args.generator_zero_stage,
                enable_hybrid_engine = self.args.enable_hybrid_engine,
                inference_tp_size = self.args.inference_tp_size,
                release_inference_cache = self.args.release_inference_cache,
                pin_parameters = (not self.args.unpin_generator_parameters),
                tp_gather_partition_size = self.args.tp_gather_partition_size, 
                max_out_tokens = self.args.max_prompt_seq_len*(self.args.per_device_training_batch_size+1)+self.args.max_answer_seq_len, # mark, different from train
                enable_tensorboard = self.args.enable_tensorboard, 
                enable_mixed_precision_lora = self.args.enable_mixed_precision_lora,
                tb_path = self.args.tensorboard_path, 
                tb_name = "gail_generator")
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_config['train_batch_size'] = self.args.per_device_training_batch_size * self.args.gradient_accumulation_steps
        print("ds_config")
        print(ds_config)
        # Model 
        generator_model = create_hf_model(
            model_class = AutoModelForCausalLM,
            model_name_or_path = generator_model_name_or_path,
            tokenizer = self.tokenizer,
            ds_config = ds_config,
            gail_training = True,
            dropout = self.args.generator_dropout)
        # DeepSpeed Engine
        print("generator engine ... ")
        generator_engine = deepspeed.init_inference(model = generator_model, 
                                                    #dtype=torch.half,
                                                
                                                    )
        #print("generator engine 2 ... ")
        log_init("gail_generator", stime = stime)
        return generator_engine , generator_model
        #return generator_model

