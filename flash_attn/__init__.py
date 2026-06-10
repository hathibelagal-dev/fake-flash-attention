import torch

__version__ = "2.7.4.post1"

# ====================== Main FA2 Functions ======================

def flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False, **kwargs):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale
    )


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
