# Project Progress: fake-flash-attention

## Status
- Initialized `GEMINI.md` with project overview and conventions.
- Created `test_shim.py` to verify functionality.
- Verified that the `flash_attn` shim correctly calls PyTorch SDPA.
- Rewrote `README.md` to provide a comprehensive explanation of the project's utility, hardware compatibility, and installation.
- Fixed `pip install` failure by updating version strings in `setup.py` and `flash_attn/__init__.py` to be PEP 440 compliant (using `.post1`).
- Renamed package to `fake-flash-attention` in `setup.py` for PyPI publication compatibility while maintaining the `flash_attn` module name for drop-in usage.
- Fixed `RuntimeError` in `flash_attn_func` caused by tensor layout mismatch. `flash-attn` uses `(batch, seqlen, nheads, head_dim)` while PyTorch SDPA expects `(batch, nheads, seqlen, head_dim)`. Added transposition logic to handle this.
- Improved shim robustness by ensuring tensor contiguity and adding warnings for unsupported features like `alibi_slopes` and `window_size`.

## Next Steps
- Consider adding more comprehensive tests for different `flash_attn` variants (e.g., varlen).
- Maintain API parity as new versions of `flash-attn` are released.
