import torch
from typing import Callable
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
from scx.keys import copy_submodules


def scx_encode_llama(model, key_generator):
    device = model.device
    for i, block in enumerate(model.model.layers):
        orig_attn = block.self_attn
        keys = key_generator.gen_keys()
        scx_attn = SCXLlamaAttention(model.config, i, keys).half().to(device)
        copy_submodules(orig_attn, scx_attn)
        block.self_attn = scx_attn


class SCXLlamaAttention(LlamaAttention):

    def __init__(self, config, layer_idx,
                 scx_keys = None):
        super().__init__(config, layer_idx)
        # SCX parameters
        self.attn_pi_left = scx_keys["attn_pi_left"]
        self.attn_pi_right = scx_keys["attn_pi_right"]
        self.attn_inv_pi_left = scx_keys["attn_inv_pi_left"]
        self.attn_inv_pi_right = scx_keys["attn_inv_pi_right"]

        self.wo_pi_left = scx_keys["wo_pi_left"]
        self.wo_inv_pi_left = scx_keys["wo_inv_pi_left"]

        self.r_embeds = scx_keys["r_embeds"]
        if self.r_embeds is not None:
            self.r_embeds_num = self.r_embeds.shape[1]
        else:
            self.r_embeds_num = 0
        self.alpha = scx_keys["alpha"]
    
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value,
        cache_position,
        **kwargs
    ):
        orig_input_shape = hidden_states.shape[:-1]
        orig_seq_len = hidden_states.shape[1]

        # SCX CPU step1
        orig_device = hidden_states.device
        hidden_states = hidden_states.to("cpu")
        if self.r_embeds_num > 0: # 如果使用冗余 embeds 做编码，则将冗余 embeds 拼接到 hidden_states 后面
            hidden_states = torch.cat([hidden_states, self.r_embeds], dim=1)
        hidden_states = hidden_states[:, self.attn_pi_left] # 在 seq_len 维度上做置换
        # if self.alpha is not None:
        #     hidden_states = hidden_states + self.alpha
        hidden_states = hidden_states.to(orig_device)
        # print("hidden_states shape: ", hidden_states.shape)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # print("hidden_shape: ", hidden_states.shape) # hidden_shape:  torch.Size([BS, SEQ_LEN, HIDDEN_DIM])

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # SCX CPU step2
        # print("query_states shape: ", query_states.shape) # bs, q_head_num, seq_len, q_head_dim
        query_states = query_states.to("cpu")
        # print("query_states shape: ", query_states.shape)
        # print("self.attn_inv_pi_left shape: ", len(self.attn_inv_pi_left))
        # print("orig_seq_len: ", orig_seq_len)
        query_states = query_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len] # 恢复正常 seq_len 顺序
        # print("query_states shape: ", query_states.shape)

        key_states = key_states.to("cpu")
        key_states = key_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len]
        
        value_states = value_states.to("cpu")
        value_states = value_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len]
        
        cos, sin = position_embeddings
        cos = cos.to("cpu")
        sin = sin.to("cpu")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states[:, :, :, self.attn_pi_right].to(orig_device) # 在 hidden_dim 维度做置换
        key_states = key_states[:, :, :, self.attn_pi_right].to(orig_device)
        value_states = value_states[:, :, :, self.attn_pi_right].to(orig_device)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # SCX CPU step3
        attn_output = attn_output.to("cpu")
        # print("attn_output shape: ", attn_output.shape) # bs, seq_len, qk_head_num, qk_head_dim
        attn_output = attn_output[:, :, :, self.attn_inv_pi_right][:, self.wo_pi_left].to(orig_device) # 正常恢复了 hidden_dim 顺序，然后再在 seq_len 维度上做置换
        # print("attn_output shape after attn_inv_pi_right and wo_pi_left: ", attn_output.shape) # bs, seq_len, qk_head_num, qk_head_dim

        attn_output = attn_output.reshape(*orig_input_shape, -1).contiguous()
        # print("attn_output shape after reshape: ", attn_output.shape) # bs, seq_len, qk_head_num, qk_head_dim
        attn_output = self.o_proj(attn_output)

        # SCX CPU step4
        attn_output = attn_output.to("cpu")
        attn_output = attn_output[:, self.wo_inv_pi_left].to(orig_device) # 恢复了 seq_len 顺序

        return attn_output, attn_weights




  