# fake-flash-attention ⚡️ (Fake)

[![PyPI version](https://img.shields.io/pypi/v/fake-flash-attention.svg)](https://pypi.org/project/fake-flash-attention/)
[![PyPI downloads](https://img.shields.io/pypi/dm/fake-flash-attention.svg)](https://pypi.org/project/fake-flash-attention/)

A drop-in, pure-Python shim for the `flash-attn` package. It redirects all FlashAttention calls to PyTorch's native `scaled_dot_product_attention` (SDPA).

## Why is this necessary?

Modern Large Language Models (LLMs) and popular libraries (like Hugging Face Transformers) often have hard-coded dependencies on the `flash-attn` package. However, the official `flash-attn` library has strict requirements:

- **NVIDIA GPU only**: Requires Turing, Ampere, Ada, or Hopper architectures (e.g., RTX 20/30/40, A100, H100).
- **No Support for Older GPUs**: Common GPUs like the **NVIDIA T4** (standard in Google Colab) or GTX 10-series cards cannot run official FlashAttention kernels.
- **No CPU Support**: Official `flash-attn` cannot be installed or run in CPU-only environments.
- **Complex Compilation**: The build process is heavy and requires specific CUDA toolkit versions.

**`fake-flash-attention` solves this by:**
1.  **API Parity**: It exports the exact same functions (e.g., `flash_attn_func`) so that libraries don't crash with an `ImportError`.
2.  **Hardware Portability**: It leverages PyTorch's `scaled_dot_product_attention`, which is highly optimized and works on **T4, older GPUs, and CPUs**.
3.  **Instant Setup**: It is a pure-Python package with no C++/CUDA compilation required.

## Installation

```bash
pip install fake-flash-attention
```

*Note: If installing from source:*
```bash
pip install .
```

## Usage

If a library or script requires `flash-attn`, install this package. Existing code will work transparently:

```python
from flash_attn import flash_attn_func
import torch

q, k, v = torch.randn(1, 12, 256, 64), torch.randn(1, 12, 256, 64), torch.randn(1, 12, 256, 64)
# This now uses PyTorch SDPA under the hood!
output = flash_attn_func(q, k, v, causal=True)
```

## Supported Features
- ✅ `flash_attn_func`
- ✅ `flash_attn_varlen_func`
- ✅ `flash_attn_qkvpacked_func` / `kvpacked`
- ✅ FlashAttention-2 API compatibility
- ✅ Device-agnostic (CPU, CUDA, MPS)
