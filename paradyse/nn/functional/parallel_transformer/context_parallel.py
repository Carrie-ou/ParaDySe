"""结合CP + FSDP
"""
from functools import partial

import torch
from torch.amp import custom_bwd, custom_fwd

from paradyse.nn.functional.parallel_transformer.metp import ring_attention_forward, ring_attention_backward
from paradyse.nn.functional.parallel_transformer.zero3 import zero_mha_function


class RingAttentionFunc(torch.autograd.Function):
    """
    Calculate attention in a flash-attention and ring-exchange style.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, sub_qkv):
        """
        Args:
            sub_qkv:  [batch_size, seq_len/p, num_heads, 3*head_size]
        Returns:
            output: [batch_size, seq_len/p, num_heads, head_size]
        """
        # save tensor for backward

        batch_size, sub_seq_length, num_heads, three_head_size = sub_qkv.shape
        ctx.sub_seq_length = sub_seq_length

        sub_q, sub_k, sub_v = torch.chunk(sub_qkv, 3, dim=-1)
        # sub_kv: [2, batch_size, num_heads, seq_len/p, head_size]
        sub_kv = torch.stack([sub_k, sub_v], dim=0)

        # output: [batch_size, sub_seq_len, num_heads, head_size]
        output, L, rng_state_list = ring_attention_forward(sub_q, sub_kv, batch_size, num_heads,
                                                           sub_seq_length,
                                                           three_head_size // 3)
        ctx.save_for_backward(sub_q, sub_kv, output, L)
        ctx.rng_state_list = rng_state_list
        ctx.num_heads = num_heads

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        sub_q, sub_kv, output, L = ctx.saved_tensors
        rng_state_list = ctx.rng_state_list

        # sub_q: [batch_size * num_heads, sub_seq_len, head_size]
        # sub_kv: [2*batch_size * num_heads, sub_seq_len, head_size]
        sub_dq, sub_dkv = ring_attention_backward(grad_output, sub_q, sub_kv, L, output, rng_state_list)

        sub_dk, sub_dv = sub_dkv

        # sub_dqkv: [batch_size * num_heads, sub_seq_len, 3*head_size]
        sub_dqkv = torch.cat([sub_dq, sub_dk, sub_dv], dim=-1)

        return sub_dqkv


def ring_attention(sub_qkv):
    """Compute the attention
    Args:
            sub_qkv:  [seq_len/p, batch_size, num_heads, 3*head_size]
        Returns:
            output: [seq_len, batch_size, hidden_size]
    """
    # output: [batch_size, seq_len / p, num_heads, head_size]
    output = RingAttentionFunc.apply(sub_qkv)
    # output: [batch_size, seq_len / p, hidden_size]
    output = output.view(output.shape[0], output.shape[1], -1)
    return output


ring_attn_zero_mha_func = partial(zero_mha_function, attn_func=ring_attention)

if __name__ == "__main__":
    # Test Code

    import os
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=0, world_size=1)

    qkv = torch.rand(1, 512, 4, 3 * 64, requires_grad=True).cuda().half()

    output = ring_attention(qkv)
    print(output.shape)

    output.sum().backward()
