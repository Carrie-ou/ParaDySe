"""结合Ulysses + FSDP

"""
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from torch import Tensor

from paradyse.communication.alltoall_4d import SeqAllToAll4D
from paradyse.nn.functional.parallel_transformer.zero3 import zero_mha_function


# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0


def ulysses_attention(
        qkv,
        *args: Any
) -> Tensor:
    """forward

    Arguments:
        qkv: [batch_size, seq_len/p, num_heads, 3*head_size]
        args: other args

    Returns:
        output: [batch_size, sub_seq_len, hidden_size]
    """
    # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!

    # scatter 2, gather 1
    # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)
    qkv = SeqAllToAll4D.apply(None, qkv, 2, 1, False)

    q, k, v = qkv.chunk(3, dim=-1)

    # k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
    # v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)

    # if softmax_scale is None:
    #     softmax_scale = q.shape[-1] ** -0.5
    context_layer = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )

    if isinstance(context_layer, tuple):
        context_layer = context_layer[0]

    # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
    # scatter 1, gather 2
    output = SeqAllToAll4D.apply(
        None, context_layer, 1, 2, False
    )

    # output: [batch_size, seq_len/p, hidden_size]
    output = output.reshape(output.shape[0], output.shape[1], -1)

    return output


ulysses_zero_mha_func = partial(zero_mha_function, attn_func=ulysses_attention)

if __name__ == "__main__":
    # Test Code

    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=0, world_size=1)

    qkv = torch.rand(1, 512, 4, 3 * 64, requires_grad=True).cuda().half()

    output = ulysses_attention(qkv)
    print(output.shape)

    output.sum().backward()
