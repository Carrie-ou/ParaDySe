"""用torch算子实现的一些函数
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def dropout_forward(x: torch.Tensor, probability=0.1, mask=None):
    if mask is None:
        mask = torch.bernoulli(x, p=probability).to(torch.int8)
    output = x * mask / (1 - probability)
    return output, mask


def dropout_backward(d_output: torch.Tensor, mask: torch.Tensor, probability=0.1):
    return d_output * mask / (1 - probability)


@torch.jit.script
# @torch.compile
def gelu_backward(grad: torch.Tensor, self: torch.Tensor):
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    kAlpha = M_SQRT1_2
    kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
    cdf = 0.5 * (1 + torch.erf(self * kAlpha))
    pdf = kBeta * torch.exp(self * self * -0.5)
    return grad * (cdf + self * pdf)


def torch_attention(qkv: torch.Tensor):
    """Using F.scaled_dot_product_attention to calculate the attention
    :arg
        qkv: [batch_size, seq_len, num_heads, 3*head_size]
    :returns
        output: [batch_size, seq_len, hidden_size]
    """
    # [batch_size, seq_len, num_heads, 3*head_size] -> [batch_size,  num_heads, seq_len, 3*head_size]
    qkv = qkv.permute(0, 2, 1, 3)
    partition_size = qkv.shape[-1] // 3

    (query_layer, key_layer, value_layer) = torch.split(qkv, partition_size, dim=-1)

    #  input shape: [b, head, seq_len, head_dim]
    # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
    #     output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer)  # , is_causal=True)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer)  # , is_causal=True)

    # output: [b, head, seq_len, head_dim] -> [b, seq_len, hidden_size]
    batch_size, num_head, seq_len, head_size = output.shape
    output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    return output
