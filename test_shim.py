import torch
from flash_attn import flash_attn_func

def test_flash_attn_shim():
    q = torch.randn(1, 4, 8, 16, dtype=torch.float16, device='cpu')
    k = torch.randn(1, 4, 8, 16, dtype=torch.float16, device='cpu')
    v = torch.randn(1, 4, 8, 16, dtype=torch.float16, device='cpu')
    
    # This should call PyTorch SDPA under the hood
    out = flash_attn_func(q, k, v, causal=True)
    
    assert out.shape == q.shape
    print("✅ Shim test passed!")

if __name__ == "__main__":
    test_flash_attn_shim()
