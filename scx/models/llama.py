import torch
from typing import Callable
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, LlamaDecoderLayer
from scx.keys import copy_submodules


def encode_llama(model, key_generator):
    device = model.device
    for i, block in enumerate(model.model.layers):
        keys = key_generator.gen_keys(layer_idx=i)

        # 先创建一个新的 decoderlayer 类
        new_block = SCXLlamaDecoderLayer(model.config, i, keys).half().to(device)
        # 把原本 decoderlayer 里的参数都复制进去
        copy_submodules(block, new_block)

        # 然后改里面的 attention
        orig_attn = block.self_attn
        scx_attn = SCXLlamaAttention(model.config, i, keys).half().to(device)
        copy_submodules(orig_attn, scx_attn)
        # 把新的 decoderlayer 里的 attention 换掉
        new_block.self_attn = scx_attn

        # 把新的 decoderlayer 替换掉原本的 decoderlayer
        model.model.layers[i] = new_block


class SCXLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, scx_keys = None):
        super().__init__(config, layer_idx)

        self.ff_pi_left = scx_keys["ff_pi_left"]
        self.ff_inv_pi_left = scx_keys["ff_inv_pi_left"]

    def forward(self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs):
        # print("SCXLlamaDecoderLayer forward kwargs: ", kwargs)
        prefill = kwargs.get("mode") == "prefill"

        # print("SCXLlamaDecoderLayer forward")
        orig_device = hidden_states.device

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        if prefill:
            # 这里需要在 CPU 里执行加法
            # 传回的 hidden_states 是 cpu 的，所以需要把 residual 也移动到 cpu 里
            residual = residual.to("cpu")
        
        hidden_states = residual + hidden_states

        if prefill:
            # seq_len 维度上做置换
            hidden_states = hidden_states[:, self.ff_pi_left]
            # 发回 gpu
            hidden_states = hidden_states.to(orig_device)

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if prefill:
            # 回到 cpu，恢复顺序 
            hidden_states = hidden_states.to("cpu")
            hidden_states = hidden_states[:, self.ff_inv_pi_left]
            hidden_states = hidden_states.to(orig_device) # 暂时保留这里，实际上可以不传回 gpu

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


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

        self.dec_attn_alp = scx_keys["dec_attn_alp"]
        self.dec_attn_pi_right = scx_keys["dec_attn_pi_right"]
        self.dec_attn_inv_pi_right = scx_keys["dec_attn_inv_pi_right"]

        self.decode_start_flag = False
        self.decode_end_flag = False
        if self.dec_attn_alp is not None:
            if self.dec_attn_pi_right is not None: # 属于 decode 阶段的开始的层
                self.decode_start_flag = True
            else: # 属于 decode 阶段结尾的层
                self.decode_end_flag = True

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value,
        cache_position,
        **kwargs
    ):
        prefill = kwargs.get("mode") == "prefill"

        gpu_kvcache = kwargs.get("gpu_kvcache")
        cpu_kvcache = kwargs.get("cpu_kvcache")

        orig_input_shape = hidden_states.shape[:-1]
        orig_seq_len = hidden_states.shape[1]
        orig_device = hidden_states.device

        if prefill: # SCX CPU step1
            hidden_states = hidden_states.to("cpu")
            if self.r_embeds_num > 0: # 如果使用冗余 embeds 做编码，则将冗余 embeds 拼接到 hidden_states 后面
                hidden_states = torch.cat([hidden_states, self.r_embeds], dim=1)
            hidden_states = hidden_states[:, self.attn_pi_left] # 在 seq_len 维度上做置换
            hidden_states = hidden_states.to(orig_device)

        else:
            if self.decode_start_flag:
                hidden_states = hidden_states.to("cpu") + self.dec_attn_alp
                hidden_states = hidden_states.to(orig_device)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if prefill: # SCX CPU step2
            query_states = query_states.to("cpu")
            query_states = query_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len] # 恢复正常 seq_len 顺序
            key_states = key_states.to("cpu")
            key_states = key_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len]
            value_states = value_states.to("cpu")
            value_states = value_states[:, :, self.attn_inv_pi_left][:, :, :orig_seq_len]
            cos = cos.to("cpu")
            sin = sin.to("cpu")

        else:
            if self.decode_start_flag: # 开始层，kvcache 操作还保留在 GPU 里
                query_states = query_states.to("cpu") # NOTE 注意，这里实际上应该减去 alp * W_q，暂时略去
                key_states = key_states.to("cpu")
                value_states = value_states.to("cpu")
                cos = cos.to("cpu")
                sin = sin.to("cpu")
            if self.decode_end_flag:
                # NOTE 实际上，q、k不 需要回到 cpu，暂时这样实现
                query_states = query_states.to("cpu")
                key_states = key_states.to("cpu")
                value_states = value_states.to("cpu")
                cos = cos.to("cpu")
                sin = sin.to("cpu")
        
        # print("query_states before rope: ", query_states[0, 0, 0, :5])
        # 这里的 query 还是保持了一致的
        # 经过 rope 后不一致。说明是 rope 这里有问题，看一下 cos 和 sin
        # print("cos", cos)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # print("kvcache update, layer_idx: ", self.layer_idx)
        # print("key_states device: ", key_states.device)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        if prefill:
            if past_key_value is not None:
                # print("*****", prefill, len(past_key_value))
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        else:
            if self.decode_start_flag or self.decode_end_flag: # decode 开始和结尾层，使用 cpu kvcache
                key_states, value_states = cpu_kvcache.update(key_states, value_states, self.layer_idx)
            else: # decode 中间层，使用 gpu kvcache
                key_states, value_states = gpu_kvcache.update(key_states, value_states, self.layer_idx)


        ### check key_states and value_states
        # print("query_states sample: ", query_states[0, 0, 0, :5])
        ### query 不一致
        # print("key_states sample: ", key_states[0, 0, 0, :10])
        # print("value_states sample: ", value_states[0, 0, 0, :10])
        ### 都是相同的！！！



        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if prefill:
            # 移动到 kvcache update 之后，这样的话，past_key_value 里的内容是 valid cache
            query_states = query_states[:, :, :, self.attn_pi_right].to(orig_device) # 在 hidden_dim 维度做置换
            key_states = key_states[:, :, :, self.attn_pi_right].to(orig_device)
            value_states = value_states[:, :, :, self.attn_pi_right].to(orig_device)

        else:
            if self.decode_start_flag:
                query_states = query_states[:, :, :, self.dec_attn_pi_right].to(orig_device)
                key_states = key_states[:, :, :, self.dec_attn_pi_right].to(orig_device)
                value_states = value_states[:, :, :, self.dec_attn_pi_right].to(orig_device)
            if self.decode_end_flag:
                query_states = query_states.to(orig_device)
                key_states = key_states.to(orig_device)
                value_states = value_states.to(orig_device)

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

        if prefill: # SCX CPU step3
            attn_output = attn_output.to("cpu")
            attn_output = attn_output[:, :, :, self.attn_inv_pi_right][:, self.wo_pi_left].to(orig_device) # 正常恢复了 hidden_dim 顺序，然后再在 seq_len 维度上做置换
        
        else:
            if self.decode_start_flag:
                attn_output = attn_output.to("cpu")
                attn_output = attn_output[:, :, :, self.dec_attn_inv_pi_right].to(orig_device) # NOTE 注意，实际上应该 Wo 在 cpu 里进行，暂时忽略

        attn_output = attn_output.reshape(*orig_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if prefill:  # SCX CPU step4
            attn_output = attn_output.to("cpu")
            attn_output = attn_output[:, self.wo_inv_pi_left] # 恢复了 seq_len 顺序，先不要传回 gpu

        return attn_output, attn_weights




  