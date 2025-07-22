from transformers import LlamaForCausalLM, LlamaConfig
import torch
from scx.keys import SCXKeyGenerator
from scx.models.llama import encode_llama

from scx.kvcache import split_kvcache_dynamic


device = "cuda:0"
num_hidden_layers = 3
seq_len = 10
hidden_dim = 4096
qk_hidden_dim = 128 # 32 qk heads
redundant_num = 0
alp = False
BS = 1
dec_steps = 5

config = LlamaConfig(vocab_size=1000, num_hidden_layers=num_hidden_layers, hidden_size=hidden_dim)
model = LlamaForCausalLM(config).eval().half().to(device)


input_ids = torch.randint(0, 1000, (BS, seq_len)).to(device)

# original prefill
with torch.no_grad():
    orig_output = model(input_ids)
    orig_logits = orig_output.logits
    orig_kvcache = orig_output.past_key_values
    # print("original logits shape: ", orig_logits.shape)
    # print("original kvcache device: ", orig_kvcache[0][0].device)

    # original next token prediction
    orig_last_token_logits = orig_logits[:, -1, :]
    orig_next_token_id = torch.argmax(orig_last_token_logits, dim=-1, keepdim=True)
    print("Predicted next token id:", orig_next_token_id.item())


    for step in range(dec_steps):
        print("Decoding step: ", step)
        # print("orig_next_token_id: ", orig_next_token_id.shape)
        orig_dec_output = model(input_ids=orig_next_token_id, past_key_values=orig_kvcache, use_cache=True)
        orig_dec_logits = orig_dec_output.logits

        orig_kvcache = orig_dec_output.past_key_values
        orig_dec_next_token_id = torch.argmax(orig_dec_logits[:, -1, :], dim=-1, keepdim=True)
        print("Predicted next token id:", orig_dec_next_token_id.item())


    print("-" * 100)
    print("SCX")
    print("-" * 100)

    # SCX key generation
    scx_key_generator = SCXKeyGenerator(seq_len, hidden_dim, qk_hidden_dim, 
                                        redundant_num=redundant_num, alp=alp, batch_size=BS,
                                        decode_start_layers=[0], decode_end_layers=[2])

    encode_llama(model, scx_key_generator)

    # prefill
    scx_output = model(input_ids, mode="prefill")
    scx_logits = scx_output.logits
    scx_kvcache = scx_output.past_key_values
    # print("scx logits shape: ", scx_logits.shape)
    # print("scx_kvcache length: ", len(scx_kvcache))

    scx_last_token_logits = scx_logits[:, -1, :]
    scx_next_token_id = torch.argmax(scx_last_token_logits, dim=-1, keepdim=True)
    print("Predicted next token id:", scx_next_token_id.item())

    max_abs_diff = torch.max(torch.abs(orig_last_token_logits - scx_last_token_logits)).item()
    print("Max absolute difference between original and SCX last token logits:", max_abs_diff)

    # 重新构建两个 kvcache，分别是放在 cpu 和 gpu 里
    gpu_kvcache, cpu_kvcache = split_kvcache_dynamic(scx_kvcache, gpu_layers=[1])

    for step in range(dec_steps):
        

        # print("gpu_kvcache length: ", len(gpu_kvcache))
        # print("cpu_kvcache length: ", len(cpu_kvcache))

        print("Decoding step: ", step)

        past_seen_tokens = seq_len + step
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=device)

        scx_dec_output = model(input_ids=scx_next_token_id, 
                               use_cache=True, 
                               mode="decode",
                               cache_position=cache_position,
                               gpu_kvcache=gpu_kvcache,
                               cpu_kvcache=cpu_kvcache)
        
        scx_dec_logits = scx_dec_output.logits
        # scx_kvcache = scx_dec_output.past_key_values # 完了，这里有问题 BUG past_key_values 是空的

        scx_dec_next_token_id = torch.argmax(scx_dec_logits[:, -1, :], dim=-1)
        print("Predicted next token id:", scx_dec_next_token_id.item())