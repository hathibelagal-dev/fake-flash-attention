import torch
from flash_attn import flash_attn_func

def test_flash_attn_shim():
    # flash-attn typically expects (batch, seqlen, nheads, d)
    # The error message "a (1356) must match the size of tensor b (257)"
    # suggests seqlen mismatch or head mismatch during broadcast.
    
    # Common layout for flash-attn: [batch, seq, heads, dim]
    q = torch.randn(1, 1356, 16, 64, dtype=torch.float16, device='cpu')
    k = torch.randn(1, 257, 16, 64, dtype=torch.float16, device='cpu')
    v = torch.randn(1, 257, 16, 64, dtype=torch.float16, device='cpu')
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # This should call PyTorch SDPA under the hood
    out = flash_attn_func(q, k, v, causal=False)
    
    assert out.shape == q.shape
    print("✅ Shim test passed!")

if __name__ == "__main__":
    test_flash_attn_shim()
