from transformers.cache_utils import DynamicCache



def split_kvcache_dynamic(kvcache: DynamicCache, gpu_layers: list, device_gpu="cuda:0", device_cpu="cpu"):
    
    gpu_kvcache = DynamicCache()
    gpu_kvcache._seen_tokens = kvcache._seen_tokens
    cpu_kvcache = DynamicCache()
    cpu_kvcache._seen_tokens = kvcache._seen_tokens
    # print("gpu_kvcache._seen_tokens: ", gpu_kvcache._seen_tokens)
    # print("cpu_kvcache._seen_tokens: ", cpu_kvcache._seen_tokens)

    for i in range(len(kvcache)):
        key, value = kvcache[i]

        if i in gpu_layers:
            gpu_kvcache.update(layer_idx=i, key_states=key.to(device_gpu), value_states=value.to(device_gpu))
        else:
            cpu_kvcache.update(layer_idx=i, key_states=key.to(device_cpu), value_states=value.to(device_cpu))

    return gpu_kvcache, cpu_kvcache