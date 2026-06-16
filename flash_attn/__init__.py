import torch
import warnings

__version__ = "2.7.4.post2"

# ====================== Main FA2 Functions ======================

def flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False, **kwargs):
    if alibi_slopes is not None:
        warnings.warn("fake-flash-attention: alibi_slopes is not supported and will be ignored. Results will diverge from real flash-attn.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # flash-attn expects (batch, seqlen, nheads, head_dim)
    # torch SDPA expects (batch, nheads, seqlen, head_dim)
    q_len, k_len = q.shape[1], k.shape[1]
    
    w_left, w_right = window_size
    if causal:
        w_right = 0
    
    attn_mask = None
    is_causal_sdpa = causal
    
    # If windowing is requested, we must use a custom mask
    if w_left >= 0 or w_right >= 0:
        # Create a boolean mask: True means keep, False means mask
        # This is a band-diagonal mask
        mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
        if causal:
            mask = torch.tril(mask) # j <= i
        if w_left >= 0:
            mask &= torch.triu(torch.ones((q_len, k_len), device=q.device, dtype=torch.bool), diagonal=-w_left) # j >= i - w_left
        if w_right >= 0:
            mask &= torch.tril(torch.ones((q_len, k_len), device=q.device, dtype=torch.bool), diagonal=w_right) # j <= i + w_right
        
        attn_mask = mask
        is_causal_sdpa = False # Mask handles causality now

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal_sdpa, scale=softmax_scale
    )

    # Transpose back to (batch, seqlen, nheads, head_dim) and ensure contiguity
    return out.transpose(1, 2).contiguous()


# FA2 common aliases
flash_attn_varlen_func = flash_attn_func
flash_attn_kvpacked_func = flash_attn_func
flash_attn_qkvpacked_func = flash_attn_func
flash_attn_varlen_kvpacked_func = flash_attn_func
flash_attn_varlen_qkvpacked_func = flash_attn_func


# ====================== Interface Module ======================
# Many libraries do: from flash_attn.flash_attn_interface import ...
class flash_attn_interface:
    flash_attn_func = flash_attn_func
    flash_attn_varlen_func = flash_attn_func
    flash_attn_kvpacked_func = flash_attn_func
    flash_attn_qkvpacked_func = flash_attn_func
    flash_attn_varlen_kvpacked_func = flash_attn_func
    flash_attn_varlen_qkvpacked_func = flash_attn_func

    # Some older calls
    flash_attn_unpadded_func = flash_attn_func
    flash_attn_unpadded_kvpacked_func = flash_attn_func
    flash_attn_unpadded_qkvpacked_func = flash_attn_func


# Expose it at package level too
from . import flash_attn_interface as interface_module


# Dummy modules some libraries check
class layers:
    class FlashAttention(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise NotImplementedError("Fake")

bert_padding = None
fused_softmax = None


print(f"✅ Fake flash-attn {__version__} (FlashAttention-2 compatible) loaded → using SDPA")
