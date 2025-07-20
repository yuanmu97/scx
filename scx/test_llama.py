from transformers import LlamaForCausalLM, LlamaConfig
import torch
from scx_llama import SCXLlamaAttention
import numpy as np
from keys import gen_scx_keys


device = "cuda:0"
seq_len = 10
hidden_dim = 4096
qk_hidden_dim = 128 # 32 qk heads
redundant_num = 0
alp = False
BS = 1

config = LlamaConfig(vocab_size=1000, num_hidden_layers=2, hidden_size=hidden_dim)
model = LlamaForCausalLM(config).half().to(device)

print(model)

input_ids = torch.randint(0, 1000, (BS, seq_len)).to(device)

# prefill
orig_output = model(input_ids).logits
print("original output shape: ", orig_output.shape)

# next token prediction
orig_last_token_logits = orig_output[:, -1, :]
orig_next_token_id = torch.argmax(orig_last_token_logits, dim=-1)
print("Predicted next token id:", orig_next_token_id.item())


def copy_submodules(src_module, tgt_module):
    for name, src_child in src_module.named_children():
        setattr(tgt_module, name, src_child)
        print(f"Copied {name}")


# 替换每层的注意力模块
for i, block in enumerate(model.model.layers):
    orig_attn = block.self_attn

    # 只用随机置换
    keys = gen_scx_keys(seq_len, hidden_dim, qk_hidden_dim, redundant_num=redundant_num, alp=alp, batch_size=BS)

    scx_attn = SCXLlamaAttention(model.config, i, keys).half().to(device)
    copy_submodules(orig_attn, scx_attn)

    # 复制参数过来
    block.self_attn = scx_attn


print("\nAfter SCX:")
# prefill
scx_output = model(input_ids).logits
print("scx output shape: ", scx_output.shape)

# next token prediction
scx_last_token_logits = scx_output[:, -1, :]
scx_next_token_id = torch.argmax(scx_last_token_logits, dim=-1)
print("Predicted next token id:", scx_next_token_id.item())


max_abs_diff = torch.max(torch.abs(orig_last_token_logits - scx_last_token_logits)).item()
print("Max absolute difference between original and SCX last token logits:", max_abs_diff)
