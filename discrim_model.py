import torch
from torch import nn

class DiscriminatorModel(nn.Module):
    
    def __init__(self, base_model, tokenizer, num_padding_at_beginning = 0, compute_fp32_loss = False, prompt_length=128):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias = False)
        else:
            self.config.n_embd = self.config.hidden_size if hasattr(
                    self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias = False)
        self.dsm_transformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        print("self.PAD_ID {}", self.PAD_ID)
        self.compute_fp32_loss = compute_fp32_loss
        self.prompt_length = prompt_length

    def gradient_checkpointing_enable(self):
        self.dsm_transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.dsm_transformer.gradient_checkpointing_disable()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, use_cache=False):
        loss = None
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask = head_mask)
        transformer_outputs = self.dsm_transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        #print("transformer_outputs.size")
        #print(transformer_outputs)
        #print(transformer_outputs[0])
        #print(transformer_outputs[0].size())
        hidden_states = transformer_outputs[0] # batch * len * hidden_size 
        rewards = torch.sigmoid(self.v_head(hidden_states).squeeze(-1)) # batch * len * hidden_size
        print("discriminator rewards.size()")
        print(rewards.size())
        #print(rewards)
        #print(rewards[:, -1])
        return rewards #[:, ]

    def forward_value(self, input_ids=None, attention_mask=None, past_key_values=None, position_ids=None, head_mask=None, inputs_embeds=None, return_value_only=False, prompt_length=0, use_cache=False):
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        #print("input_ids.size()")
        #print(input_ids.size())
        #print("attention_mask")
        #print(attention_mask.size())
        transformer_outputs = self.dsm_transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            the_scores = []
            for i in range(bs):
                input_id, value = input_ids[i], values[i]
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
                the_scores.append(value[c_ind - 1])
                #the_scores.append(value[-1])
        return {"values": values, "scores": torch.stack(the_scores)}



