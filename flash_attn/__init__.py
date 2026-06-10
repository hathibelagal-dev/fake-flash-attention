import torch

__version__ = "2.7.4.post1"

# ====================== Main FA2 Functions ======================

import warnings

def flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False, **kwargs):
    if alibi_slopes is not None:
        warnings.warn("fake-flash-attention: alibi_slopes is not supported and will be ignored. Results will diverge from real flash-attn.")
    if window_size != (-1, -1):
        warnings.warn(f"fake-flash-attention: window_size {window_size} is not supported and will be ignored. Results will diverge from real flash-attn.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # flash-attn expects (batch, seqlen, nheads, head_dim)
    # torch SDPA expects (batch, nheads, seqlen, head_dim)
    # We use contiguous() to ensure the SDPA backend (which might be a C++ kernel) 
    # handles the strides correctly.
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale
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
