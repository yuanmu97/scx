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
from scx.scx_llama import scx_encode_llama
from scx.keys import SCXKeyGenerator

# Initialize model
config = LlamaConfig(vocab_size=1000, num_hidden_layers=2, hidden_size=4096)
model = LlamaForCausalLM(config).half().to("cuda")

# Create SCX key generator
key_generator = SCXKeyGenerator(
    seq_len=10,
    hidden_dim=4096,
    qk_hidden_dim=128,
    redundant_num=0,
    batch_size=1
)

# Apply SCX encoding to the model
scx_encode_llama(model, key_generator)

# Use the encoded model for inference
input_ids = torch.randint(0, 1000, (1, 10)).to("cuda")
output = model(input_ids)
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

## Examples

See `scx/test_llama.py` for a complete example demonstrating SCX with Llama models.

## License

[Add your license information here]

## Citation

[Add citation information if applicable]
