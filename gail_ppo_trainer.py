
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from torch.nn.utils.rnn import pad_sequence

from gail_utils import print_rank_0


def print_all_ranks(tag, value, rank):
    #world_size = torch.distributed.get_world_size()
    #all_tensor = torch.zeros(world_size, dtype=torch.float32).to(get_accelerator().current_device_name())
    #all_tensor[rank] = value
    #torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    #print_rank_0(f'{tag} {all_tensor}', rank)
    pass

def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            #should_gather = hasattr(param, 'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            #with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                total += float(param.float().norm())
    return total

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedGailPPOTrainer():

    def __init__(self, gail_engine, args):
        self.gail_engine = gail_engine

        self.generator_model = self.gail_engine.generator_model
        self.generator_optim = self.gail_engine.generator_optim
        self.generator_lr_scheduler = self.gail_engine.generator_lr_scheduler

        self.ref_generator_model = self.gail_engine.ref_generator_model

        self.critic_model = self.gail_engine.critic_model
        self.critic_optim = self.gail_engine.critic_optim
        self.critic_lr_scheduler = self.gail_engine.critic_lr_scheduler

        self.discriminator_model = self.gail_engine.discriminator_model
        self.discriminator_optim = self.gail_engine.discriminator_optim
        self.discriminator_lr_scheduler = self.gail_engine.discriminator_lr_scheduler

        self.tokenizer = self.gail_engine.tokenizer

        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.max_prompt_seq_len = args.max_prompt_seq_len
        self.end_of_conversation_token_id = self.tokenizer(args.end_of_conversation_token)['input_ids'][-1]
        self.compute_fp32_loss = self.args.compute_fp32_loss
        # 
        self.last_generated_experience = None
        self.pad_token_id = self.tokenizer.pad_token_id
        #
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.generate_time = 0.0

        # loss defination
        self.discriminator_criterion = nn.BCELoss()

    def _generate_sequence(self, prompts, mask, step):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        if self.generator_model.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict(do_sample=True)
        #print("prompts")
        #print(prompts)
        #print(mask)
        #print(max_min_length)
        #print(self.tokenizer)
        #print(self.tokenizer.pad_token_id)
        with torch.no_grad():
            seq = self.generator_model.generate(prompts, attention_mask=mask, max_length=max_min_length, pad_token_id = self.tokenizer.pad_token_id, **kwargs)
            seq = pad_sequence([f for f in seq],
                                padding_value=self.pad_token_id,
                                batch_first=True)

        pad_length = self.max_prompt_seq_len + self.max_answer_seq_len - seq.shape[1]
        if pad_length > 0:
            #print("seq")
            #print(seq)
            #print(pad_length)
            seq = F.pad(seq, 
                        pad=(0, pad_length), 
                        mode='constant', 
                        value=self.pad_token_id)

        # Filter out seq with no answers (or very shot).
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        #out_seq = []
        #for i in range(batch_size):
        #    if valid_ans_len[i] <= 1:
        #        continue
        #    else:
        #        out_seq.append(seq[i:i+1])
        #if not out_seq:
        #    return None
        #out_seq = torch.cat(out_seq, dim=0)
        print("_generate_sequence result")
        print("batch_size: " , batch_size)
        print("prompt_length: ", prompt_length)
        print("ans: ", ans)
        print("ans shape: ", ans.shape)
        print("seq: ", seq)
        print("seq shape: ", seq.shape)
        print("valid_ans_len: ", valid_ans_len)
        #print("out_seq: ", out_seq)
        print("seq 0 decode: ", self.tokenizer.decode(seq[0], skip_special_tokens=True))
        print("ans 0 decode: ", self.tokenizer.decode(ans[0], skip_special_tokens=True))
        print("end _gnerea")
        return seq, valid_ans_len , ans #out_seq

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        prompt_length = prompts.shape[1]
        seq, valid_ans_len, ans = self._generate_sequence(prompts, mask, step)
        #print("seq ... ")
        #print(seq)
        #print(seq.size())
        generate_end = time.time()
        if seq is None:
            print("seq is None ... ")
        self.train()
        
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        #print("attention_mask.size()")
        #print(attention_mask.size())
        #print(attention_mask)
        #discrim_rewards, critic_values = [] , []
        with torch.no_grad():
            output = self.generator_model(seq, attention_mask = attention_mask)
            #print("output ... ")
            #print(len(output))
            #print(output.logits.size())
            #print(len(output.past_key_values))
            #print(len(output.past_key_values[0]))
            #print(output.past_key_values[0][0].size())
            #output_ref
            #output[ : , pro]
            #print("discrim_scalar.size()")
            output_ref = self.ref_generator_model(seq, attention_mask = attention_mask)
            #print("output ref ... ")
            #print(len(output_ref))
            #print(output_ref.logits.size())
            #reward_score = self.discriminator.forward_value(seq, attention_mask,
            #                prompt_length=self.prompt_length)["scores"].detach()
            #values = self.critic_model.forward_value(seq, attention_mask,
            #                return_value_only=True).detach()[:, :-1]
            ''' discrim_scalar's size is batch * 1
            '''
                
            critic_values = self.critic_model.forward_value(seq,
                                            attention_mask,
                                            return_value_only=True).detach()[:, :-1]
            discrim_rewards = self.discriminator_model.forward_value(seq,
                            attention_mask,
                            prompt_length=self.prompt_length)["scores"].detach()

        logits, logits_ref = output.logits, output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)
        self.generate_time = generate_end - generate_start
        #print("critic_values and discrim_rewards")
        #print(critic_values.size())
        #print(discrim_rewards.size())
        #print(torch.cat(critic_values, 0).size())
        #print(torch.cat(discrim_rewards,0).size())
        res = {"prompts": prompts, # batch * prompt_len
                "logprobs": gather_log_probs(logits[:,prompt_length:,:], seq[:,prompt_length:]), # batch * (2*prompt_len) - 1
                "ref_logprobs": gather_log_probs(logits_ref[:,prompt_length:,:], seq[:,prompt_length:]), # batch* (2*prompt_len) - 1
                #"value": torch.transpose(torch.cat(critic_values, 0), 0, 1), # batch * prompt_len
                "value": critic_values,
                #"rewards": torch.transpose(torch.cat(discrim_rewards, 0), 0, 1), # batch * prompt_len
                "rewards": discrim_rewards,
                "input_ids": seq, # batch * (2*prompt_len)
                "answers": ans,
                "attention_mask": attention_mask} # batch * (2*prompt_len)
        return res

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        #print("log_probs size: ", log_probs.size())
        #print("ref log_probs size: ", ref_log_probs.size())
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1 # prompt_length - 1
        ends = start + action_mask[:, start:].sum(1) + 1 # 2 * prompt_length
        reward_clip = torch.clamp(reward_score, 
                        -self.clip_reward_value, 
                        self.clip_reward_value)
        batch_size = log_probs.shape[0]
        #print("rewards reward_clip size")
        #print(rewards.size())
        #print(reward_clip.size())
        #print("start: ", start, " ends: ", ends)
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
            #rewards[j, :][-1] += reward_clip[j]
        return rewards
    
    def train_discriminator(self, expert_data, generation_data, device):
        #print("discriminator ... ")
        #print(expert_data["tokens"].size())
        #print(expert_data["tokens"])
        #print(expert_data["token_att_mask"].size())
        #print(expert_data["token_att_mask"])
        #print(generation_data["input_ids"].size())
        #print(generation_data["input_ids"])
        #print(generation_data["attention_mask"].size())
        #print(generation_data["attention_mask"])
        temp_exp_token = torch.cat([generation_data["input_ids"][:self.max_prompt_seq_len] , expert_data["tokens"]], dim=-1)
        temp_exp_att = torch.cat([generation_data["attention_mask"][:self.max_prompt_seq_len], expert_data["token_att_mask"]], dim=-1)
        exp_q_value = self.discriminator_model(temp_exp_token, attention_mask = temp_exp_att)[:, self.max_answer_seq_len:] 
        #print(exp_q_value.dtype)
        #print(exp_q_value.size())
        #print(exp_q_value)
        #print("zeros")
        #print(torch.zeros((exp_q_value.shape[0], exp_q_value.shape[1]), device=device))
        gen_q_value = self.discriminator_model(generation_data["input_ids"], attention_mask = generation_data["attention_mask"])[:, self.max_answer_seq_len:]
        loss_exp = self.discriminator_criterion(exp_q_value.type(torch.float32), torch.zeros((exp_q_value.shape[0], exp_q_value.shape[1]), device=device))

        loss_gen = self.discriminator_criterion(gen_q_value.type(torch.float32), torch.ones((gen_q_value.shape[0], gen_q_value.shape[1]), device=device))
        loss_discriminator = loss_exp + loss_gen
        #print("loss ... ")
        #print(loss_exp.data.cpu().numpy())
        #print(loss_gen.data.cpu().numpy())
        #print(loss_discriminator.cpu().detach().numpy())
        #self.discriminator_model.backward(loss_discriminator)
        loss_discriminator.backward()
        self.discriminator_optim.step()
        self.discriminator_lr_scheduler.step()
        self.discriminator_optim.zero_grad()
        #self.discriminator_model.step()

        return loss_exp, loss_gen, loss_discriminator

    def train_gail(self, inputs):
        # train gail-kd
        ## process the old outputs
        prompts = inputs["prompts"]
        log_probs = inputs["logprobs"]
        ref_log_probs = inputs["ref_logprobs"]
        reward_score = inputs["rewards"]
        values = inputs["value"]
        attention_mask = inputs["attention_mask"]
        seq = inputs["input_ids"]
        ans = inputs["answers"]
        
        prompt_length = self.max_prompt_seq_len

        #print("prompts size: ", prompts.size())
        #print("log_probs size: ", log_probs.size())
        #print("ref_log_probs size: ", ref_log_probs.size())
        #print("reward_score size: ", reward_score.size())
        #print("values size: ", values.size())
        #print("attention_mask size: ", attention_mask.size())
        #print("seq size: ", seq.size())

        start = prompts.size()[-1] - 1 # prompt - 1
        #print("start : " , start)
        action_mask = attention_mask[:, 1:] # 2 * prompt_length - 1
        #action_mask = attention_mask[:, prompt_length:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, 
                                        log_probs,
                                        ref_log_probs,
                                        reward_score,
                                        action_mask)
            #print("old_rewards size: ", old_rewards.size())
            ends = start + action_mask[:, start:].sum(1) + 1
            #print("ends: ", ends)
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                #old_rewards[i, :] = 0
                old_values[i, ends[i]:] = 0
                #old_values[i, :] = 0
            advantages, returns = self.get_advantages_and_returns(
                    old_values, old_rewards, start)
            #print("advantages size: ", advantages.size())
            #print(advantages)
            #print("returns size: ", returns.size())
            #print(returns)
        ## process the new outputs
        batch = {"input_ids": seq, "attention_mask": attention_mask}
        generator_prob = self.generator_model(**batch, use_cache=False).logits
        generator_log_prob = gather_log_probs(generator_prob[:,:-1,:], seq[:,1:])
        #print("generator_prob and log_prob size: ")
        #print(generator_prob.size())
        #print(generator_log_prob.size())
        generator_loss = self.generator_loss_fn(generator_log_prob[:,start:],
                                                log_probs[:,start:],
                                                advantages,
                                                action_mask[:,start:])
        #self.generator_model.backward(generator_loss)
        generator_loss.backward()
        self.generator_optim.step()
        self.generator_lr_scheduler.step()
        self.generator_optim.zero_grad()

        #if not self.args.align_overflow:
        #    self.generator_model.step()

        value = self.critic_model.forward_value(**batch,
                                            return_value_only=True,
                                            use_cache=False)[:, :-1]
        #print("value size")
        #print(value.size())
        critic_loss = self.critic_loss_fn(value[:,start:],
                                        old_values[:,start:],
                                        returns,
                                        action_mask[:,start:])
        #self.critic_model.backward(critic_loss)
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_lr_scheduler.step()
        self.critic_optim.zero_grad()
        '''
        if self.args.align_overflow:
            generator_overflow = self.generator_model.optimizer.check_overflow(external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(external=True)
            rank = torch.distributed.get_rank()
            if generator_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
            elif not generator_overflow and critic_overflow:
                self.generator_model.optimizer.skip_step = True
            elif generator_overflow and critic_overflow:
                pass
            self.generator_model.step()
        '''
        #self.generator_model.step()
        #self.critic_model.step()

        return generator_loss, critic_loss

    def generator_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        gen_loss1 = -advantages * ratio
        gen_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                                    1.0 + self.cliprange)
        gen_loss = torch.sum(torch.max(gen_loss1, gen_loss2) * mask) / mask.sum()
        return gen_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        values_clipped = torch.clamp(values, old_values - self.cliprange_value,
                                             old_values + self.cliprange_value,)
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped_float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def discriminator_loss_fn(self, ):
        pass

    def get_advantages_and_returns(self, values, rewards, start):
        last_gae_lambda = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for l in reversed(range(start, length)):
            next_values = values[:, l + 1] if l < length - 1 else 0.0
            delta = rewards[:, l] + self.gamma * next_values - values[: , l]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns 

    def train(self):
        self.generator_model.train()
        self.critic_model.train()
        self.discriminator_model.train()

    def eval(self):
        self.generator_model.eval()
        self.critic_model.eval()
        self.discriminator_model.eval()

class DeepSpeedGailEvaluate():

    def __init__(self, gail_engine, args):
        self.gail_engine = gail_engine
        self.generator_model = self.gail_engine.generator
        self.tokenizer = self.gail_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.max_prompt_seq_len = args.max_prompt_seq_len * args.per_device_generation_batch_size # mark, difference from trainer
        self.end_of_conversation_token_id = self.tokenizer(args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.generator_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss
        
        # 
        self.last_generated_experience = None

    def _generate_sequence(self, prompts, mask, step):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        kwargs = dict()
        with torch.no_grad():
            seq = self.generator_model.module.generate(prompts, attention_mask=mask, max_length=max_min_length, pad_token_id = self.tokenizer.pad_token_id, synced_gpus=self.z3_enabled, **kwargs)
        # Filter out seq with no answers (or very shot).
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i+1])
        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None
        out_seq = torch.cat(out_seq, dim=0)
        return out_seq, valid_ans_len

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        prompt_length = prompts.shape[1]
        seq, valid_ans_len = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()
        
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.generator_model(seq, attention_mask = attention_mask)

        self.generate_time = generate_end - generate_start
        res = {"prompts": prompts, # batch * prompt_len
                "input_ids": seq, # batch * (2*prompt_len)
                "attention_mask": attention_mask} # batch * (2*prompt_len)
        return res

    def train(self):
        self.generator_model.train()

    def eval(self):
        self.generator_model.eval()


