# Project Overview
`fake-flash-attention` is a compatibility shim for the `flash-attn` library. It provides the same function signatures as the real `flash-attn` package but redirects the implementation to PyTorch's native `scaled_dot_product_attention` (SDPA).

This project is intended for:
- Running models designed for FlashAttention on hardware that doesn't support it (e.g., NVIDIA T4, older GTX cards, or CPU-only environments).
- Development environments where the full `flash-attn` build is too heavy or fails to compile.

## Core Technologies
- **Python**: 3.8+
- **PyTorch**: Required for SDPA support.

## Architecture
The package mocks the `flash_attn` namespace. Key functions like `flash_attn_func`, `flash_attn_varlen_func`, and various kv-packed/qkv-packed variants are aliased to a single function that wraps `torch.nn.functional.scaled_dot_product_attention`.

# Building and Running

## Installation
To install from PyPI:
```bash
pip install fake-flash-attention
```

To install in editable mode from source:
```bash
pip install -e .
```

## Running Tests
You can verify the shim is working by running the provided test script:
```bash
python3 test_shim.py
```

# Development Conventions
- **Compatibility**: The goal is to maintain API parity with `flash-attn` (specifically FlashAttention-2).
- **Simplicity**: Implementation should remain as simple as possible, delegating all heavy lifting to PyTorch SDPA.
- **No C++**: This package is pure Python to ensure maximum portability.
