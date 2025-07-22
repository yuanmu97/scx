# SCX: Stateless KV-Cache Encoding for Cloud-Scale Confidential Transformer Serving

SCX is a novel approach for confidential transformer inference that uses stateless encoding to protect sensitive information during cloud-based model serving.

## Overview

SCX provides confidentiality for transformer models by applying random permutations and encoding to the model's internal states, making it difficult to extract sensitive information from intermediate computations.

## Key Features

- **Transformer Compatibility**: Works with popular transformer architectures like Llama
- **Minimal Overhead**: Efficient implementation with minimal computational overhead
- **Configurable Security**: Adjustable encoding parameters for different security requirements

## Quick Start

### Installation

```bash
pip install torch transformers
```

### Basic Usage with Llama

```python
from transformers import LlamaForCausalLM, LlamaConfig
import torch
from scx.keys import SCXKeyGenerator
from scx.models.llama import encode_llama
from scx.kvcache import split_kvcache_dynamic

# Initialize model
config = LlamaConfig(vocab_size=1000, num_hidden_layers=3, hidden_size=4096)
model = LlamaForCausalLM(config).eval().half().to("cuda")

# Create SCX key generator
key_generator = SCXKeyGenerator(
    seq_len=10,
    hidden_dim=4096,
    qk_hidden_dim=128,
    redundant_num=0,
    alp=False,
    batch_size=1,
    decode_start_layers=[0],
    decode_end_layers=[2]
)

# Apply SCX encoding to the model
encode_llama(model, key_generator)

# Prefill phase
input_ids = torch.randint(0, 1000, (1, 10)).to("cuda")
output = model(input_ids, mode="prefill")
logits = output.logits
kvcache = output.past_key_values

# Split KV cache for GPU/CPU distribution
gpu_kvcache, cpu_kvcache = split_kvcache_dynamic(kvcache, gpu_layers=[1])

# Decode phase
for step in range(5):
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    past_seen_tokens = 10 + step
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device="cuda")
    
    output = model(
        input_ids=next_token_id,
        use_cache=True,
        mode="decode",
        cache_position=cache_position,
        gpu_kvcache=gpu_kvcache,
        cpu_kvcache=cpu_kvcache
    )
    logits = output.logits
```

## Advanced Usage

### Model Comparison Test

The framework includes comprehensive testing to compare original and SCX-encoded model outputs:

```python
# Original model inference
with torch.no_grad():
    orig_output = model(input_ids)
    orig_logits = orig_output.logits
    orig_kvcache = orig_output.past_key_values
    
    # Compare with SCX-encoded model
    scx_output = model(input_ids, mode="prefill")
    scx_logits = scx_output.logits
    
    # Verify output consistency
    max_diff = torch.max(torch.abs(orig_logits - scx_logits)).item()
    print(f"Maximum difference: {max_diff}")
```
## How It Works

SCX applies multiple layers of encoding:

1. **Sequence Permutation**: Reorders input sequences using random permutations
2. **Hidden Dimension Permutation**: Shuffles hidden dimensions in attention computations
3. **Redundant Embeddings**: Optionally adds noise embeddings for additional security
4. **Inverse Operations**: Applies inverse permutations to maintain model functionality

## Configuration Parameters

- `redundant_num`: Number of redundant embeddings (0 for no redundancy)
- `batch_size`: Batch size for processing
- `alp`: Whether to use additive noise
- `decode_start_layers`: Starting layers for decode phase
- `decode_end_layers`: Ending layers for decode phase

## Examples

See the following test files for complete examples:

- `tests/llama.test.py`: Comprehensive test with prefill/decode phases and cache management

## License

[Add your license information here]

## Citation

[Add citation information if applicable]
