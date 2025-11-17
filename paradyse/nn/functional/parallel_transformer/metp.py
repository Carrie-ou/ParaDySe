#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import distributed as dist
from torch.amp import custom_bwd, custom_fwd

from paradyse.communication.async_ring import async_ring_forward
from paradyse.nn.functional.flash_attention import FlashAttnKVPackedFunc
from paradyse.nn.functional.functional import (dropout_backward,
                                               dropout_forward,
                                               gelu_backward)
from paradyse.utils.global_buffer import get_global_memory_buffer


def ring_attention_forward(sub_q, sub_kv, batch_size, num_attention_heads, sub_seq_length, attention_head_size,
                           no_comm=False, async_op=True, skip_comm=False):
    """

    Args:
        sub_q:  (b, seq_len/p, head_num/p or head_num, head_size)
        sub_kv:  (2, b, seq_len/p, head_num/p or head_num, head_size)
        batch_size: batch_size
        num_attention_heads: num_heads
        sub_seq_length: sub_seq_len = sequence_length / parallel_size
        attention_head_size: attention head size
    Returns:
        output: [batch_size, sub_seq_len, num_heads, head_size]

    """
    # save tensor for backward
    # (b, seq_len/p, head_num/p, 3 * head_size) -> (3, b, seq_len/p, head_num/p , head_size)
    kwargs = {"dtype": sub_q.dtype, "device": sub_q.device}

    max_row_a_shape = [batch_size, num_attention_heads, sub_seq_length]
    output_shape = [batch_size, sub_seq_length, num_attention_heads, attention_head_size]

    # 用fp32来存储softmax_row_sum，避免溢出
    softmax_row_sum = torch.zeros(*max_row_a_shape, **kwargs).float()

    output = torch.zeros(*output_shape, **kwargs).float()
    local_world_size = dist.get_world_size() if not no_comm else 1
    if no_comm:
        skip_comm = True

    sub_kv_list = [get_global_memory_buffer().get_tensor(sub_kv.shape, sub_kv.dtype,
                                                         f"ring_attention_{i}") for i in range(2)]
    sub_kv_list[0].copy_(sub_kv)
    op = [None, None]
    cur = 0

    # compute attention in ring-all-reduce style
    rng_state_list = []
    for i in range(local_world_size):
        # Overlap outer-loop communication with inner-loop computation
        if op[cur] is not None:
            for req in op[cur]:
                req.wait()

        if i < local_world_size - 1:
            # sub_kv_list[1 - cur] = torch.empty_like(sub_kv_list[cur])
            op[1 - cur] = async_ring_forward(sub_kv_list[cur], sub_kv_list[1 - cur]) if not skip_comm else None
            if op[1 - cur] is not None and not async_op:
                for req in op[1 - cur]:
                    req.wait()
        # compute attention
        out, softmax_lse, _, rng_state = FlashAttnKVPackedFunc.forward(sub_q, sub_kv_list[cur])
        rng_state_list.append(rng_state)
        softmax_row_sum += torch.exp(softmax_lse)
        output += out * torch.exp(softmax_lse.unsqueeze(-1).transpose(1, 2))

        cur = 1 - cur

    L = torch.log(softmax_row_sum)


    output /= softmax_row_sum.unsqueeze(-1).transpose(1, 2)
    output = output.half()
    # Note: If we want to save some time, we can move sub_kv saving to the begin of this function,
    # but it will cost one sub_kv more memory.

    return output, L, rng_state_list


def ring_attention_backward(grad_output, sub_q, sub_kv, L, output, rng_state_list, no_comm=False, async_op=True,
                            skip_comm=False):
    """
    Args:
        grad_output: [batch_size * num_heads, sub_seq_len, head_size]
        sub_q:  (b, seq_len/p, head_num/p or head_num, head_size)
        sub_kv:  (2, b, seq_len/p, head_num/p or head_num, head_size)
        L:
        output:
    Returns:
        sub_dq
        sub_dkv

    Returns:

    """
    local_world_size = dist.get_world_size() if not no_comm else 1
    if no_comm:
        skip_comm = True
    sub_kv_dkv = torch.zeros((2, *sub_kv.shape), dtype=sub_q.dtype, device=sub_q.device)
    sub_kv_dkv[0] = sub_kv
    sub_dq = torch.zeros_like(sub_q)
    D_i = torch.sum(grad_output * output, dim=2, keepdim=True)
    sub_kv_dkv_list = [sub_kv_dkv, torch.zeros_like(sub_kv_dkv)]
    op = [None, None]
    cur = 0

    # compute attention in ring-all-reduce style
    for i in range(local_world_size):
        # Overlap outer-loop communication with inner-loop computation

        # if op[cur] is not None:
        #     for req in op[cur]:
        #         req.wait()
        if i == 0:
            # Send kv only
            op[1 - cur] = async_ring_forward(sub_kv_dkv_list[cur][0],
                                             sub_kv_dkv_list[1 - cur][0]) if not skip_comm else None
        elif i == local_world_size - 1:
            # Send dkv only
            op[1 - cur] = async_ring_forward(sub_kv_dkv_list[cur][1],
                                             sub_kv_dkv_list[1 - cur][1]) if not skip_comm else None
        else:
            # Send k_i v_i and dk_{i-1} dv_{i-1}
            op[1 - cur] = async_ring_forward(sub_kv_dkv_list[cur], sub_kv_dkv_list[1 - cur]) if not skip_comm else None
        if op[1 - cur] is not None and not async_op:
            for req in op[1 - cur]:
                req.wait()
        dq_, dkv_ = FlashAttnKVPackedFunc.backward(grad_output, sub_q, sub_kv_dkv_list[cur][0], output, L,
                                                   rng_state_list[i])
        sub_dq += dq_

        if op[1 - cur] is not None:
            for req in op[1 - cur]:
                req.wait()
        sub_kv_dkv_list[1 - cur][1] += dkv_

        cur = 1 - cur

    ops = async_ring_forward(sub_kv_dkv_list[cur][1], sub_kv_dkv_list[1 - cur][1]) if not skip_comm else None
    if ops:
        for op in ops:
            op.wait()

    ##[seq_len / p, batch_size, num_heads, 3 * head_size]

    return sub_dq, sub_kv_dkv_list[1 - cur][1]


class METPMultiHeadSelfAttentionFunc(torch.autograd.Function):
    """METP multihead self attention contains two loop.
    Inner loop do RingAttention, Outer loop do QKV in projection and out projection.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, sub_hidden_states, sub_projection_w, batch_size, hidden_size, sub_seq_length,
                num_attention_heads, config):
        """

        Args:
            sub_hidden_states:  [batch_size, sub_seq_len/p, hidden_size] # seq_length first layout.
            sub_projection_w: [4*hidden_size/p, hidden_size], concat of W1^T and W2
            batch_size: batch_size
            num_attention_heads: num_heads. head_size = hidden_size / num_attention_heads.
            sub_seq_length: sub_seq_len = sequence_length / parallel_size
            hidden_size: hidden_size
        Returns:
            output : [batch_size, sub_seq_len/p, hidden_size]

        """
        # save tensor for backward
        ctx.batch_size = batch_size
        ctx.hidden_size = hidden_size
        ctx.sub_seq_length = sub_seq_length
        ctx.num_attention_heads = num_attention_heads
        ctx.config = config

        output = torch.zeros(
            batch_size,
            sub_seq_length,
            hidden_size,
            dtype=sub_hidden_states.dtype,
            device=sub_hidden_states.device,
        )
        if config.no_comm:
            config.skip_comm = True
        local_world_size = dist.get_world_size() if not config.no_comm else 1
        rank = dist.get_rank()
        sub_hidden_size = hidden_size // local_world_size
        head_size = hidden_size // num_attention_heads
        sub_head_num = num_attention_heads // local_world_size

        # use circular buffer to store the communication tensor
        flatten_sub_w1w2 = sub_projection_w.flatten()
        buffer_list = [get_global_memory_buffer().get_tensor(flatten_sub_w1w2.shape,
                                                             flatten_sub_w1w2.dtype,
                                                             f"metp_broadcast_{i}") for i in range(2)]
        buffer_list[0].copy_(flatten_sub_w1w2)

        # Start broadcasting W_0
        op = [None, None]
        if not config.skip_comm:
            if config.async_op:
                op[0] = dist.broadcast(buffer_list[0], src=0, async_op=True)
            else:
                dist.broadcast(buffer_list[0], src=0)

        w1_numel = 3 * sub_hidden_size * hidden_size
        w2_numel = sub_hidden_size * hidden_size

        sub_w1_list = [buffer_list[i][:w1_numel].view(3 * sub_hidden_size, hidden_size) for i in range(2)]
        sub_w2_list = [buffer_list[i][w1_numel:].view(sub_hidden_size, hidden_size) for i in
                       range(2)]

        cur = 0
        L_list = []
        attn_output_list = []
        rng_list_list = []

        for i in range(local_world_size):
            # Overlap outer-loop communication of i+1 with inner-loop computation
            if i == rank - 1:
                buffer_list[1 - cur].copy_(flatten_sub_w1w2)
            if i < local_world_size - 1:
                if not config.skip_comm:
                    op[1 - cur] = dist.broadcast(buffer_list[1 - cur], src=i + 1, async_op=config.async_op)
            if op[cur] is not None:
                op[cur].wait()

            # Split into in projection weight andn out projection weight
            sub_qkv_proj = sub_w1_list[cur]
            # (hidden_size/p, hidden_size)
            sub_out_proj = sub_w2_list[cur]

            # Prepare sub qkv for Inner Loop
            # (b, seq_len/p, hidden_size) @ (hidden_size, hidden_size/p) -> (b, seq_len/p, 3*hidden_size/p)
            sub_qkv = torch.matmul(sub_hidden_states, sub_qkv_proj.t())

            sub_qkv = sub_qkv.view(batch_size, sub_seq_length, sub_head_num, 3 * head_size)

            sub_q, sub_k, sub_v = torch.chunk(sub_qkv, 3, dim=-1)
            # sub_kv: [2, batch_size, num_heads, seq_len/p, head_size]
            sub_kv = torch.stack([sub_k, sub_v], dim=0)

            # Inner Loop Attention Computation ...
            # sub_attn shape: [batch_size, sub_seq_len, sub_head_num, head_size]
            sub_attn, L, rng_state_list = ring_attention_forward(sub_q, sub_kv, batch_size, sub_head_num,
                                                                 sub_seq_length,
                                                                 head_size, no_comm=config.no_comm,
                                                                 async_op=config.async_op, skip_comm=config.skip_comm)
            rng_list_list.append(rng_state_list)
            ### Output Projections
            out_proj_input = sub_attn.reshape(batch_size, sub_seq_length, -1)
            output += torch.matmul(out_proj_input, sub_out_proj)  # [b, s/p, h/p] @ [h/p, h] -> [b, s/p, h]
            L_list.append(L)
            attn_output_list.append(sub_attn)

            cur = 1 - cur
        ctx.save_for_backward(sub_hidden_states, flatten_sub_w1w2)
        ctx.L_list = L_list
        ctx.attn_output_list = attn_output_list
        ctx.rng_list_list = rng_list_list

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        sub_hidden_states, flatten_sub_w1w2 = ctx.saved_tensors
        batch_size = ctx.batch_size
        hidden_size = ctx.hidden_size
        sub_seq_length = ctx.sub_seq_length
        num_attention_heads = ctx.num_attention_heads
        L_list = ctx.L_list
        rng_list_list = ctx.rng_list_list
        attn_output_list = ctx.attn_output_list
        config = ctx.config
        # no_comm = ctx.no_comm
        # async_op = ctx.async_op
        # use_flash = ctx.use_flash
        # skip_comm = ctx.skip_comm

        local_world_size = dist.get_world_size() if not config.no_comm else 1
        rank = dist.get_rank()
        sub_hidden_size = hidden_size // local_world_size
        head_size = hidden_size // num_attention_heads

        # use circular buffer to store the communication tensor

        broadcast_buffer_list = [get_global_memory_buffer().get_tensor(flatten_sub_w1w2.shape,
                                                                       flatten_sub_w1w2.dtype,
                                                                       f"metp_broadcast_{i}") for i in range(2)]

        # op = [None] * 2
        broadcast_buffer_list[0].copy_(flatten_sub_w1w2)
        op = [dist.broadcast(broadcast_buffer_list[0], src=0,
                             async_op=config.async_op), None] if not config.skip_comm else [None, None]

        w1_numel = 3 * sub_hidden_size * hidden_size
        w2_numel = sub_hidden_size * hidden_size
        sub_w1_list = [broadcast_buffer_list[i][:w1_numel].reshape(3 * sub_hidden_size, hidden_size) for i in range(2)]
        sub_w2_list = [broadcast_buffer_list[i][w1_numel:].reshape(sub_hidden_size, hidden_size) for
                       i in range(2)]

        reduce_buffer_list = [get_global_memory_buffer().get_tensor(flatten_sub_w1w2.shape,
                                                                    flatten_sub_w1w2.dtype,
                                                                    f"metp_reduce_{i}") for i in range(2)]
        reduce_op_list = [None, None]
        # reduce_op = [None, None]
        cur = 0
        d_input = torch.zeros_like(sub_hidden_states).reshape(-1, hidden_size)
        d_sub_w_result = torch.empty_like(flatten_sub_w1w2)

        grad_output = grad_output.reshape([sub_seq_length * batch_size, hidden_size])

        sub_head_num = num_attention_heads // local_world_size

        for i in range(local_world_size):
            # Overlap outer-loop communication with inner-loop computation
            if i == rank - 1:
                broadcast_buffer_list[1 - cur].copy_(flatten_sub_w1w2)
            if i < local_world_size - 1:
                op[1 - cur] = dist.broadcast(broadcast_buffer_list[1 - cur], src=i + 1,
                                             async_op=config.async_op) if not config.skip_comm else None
            if op[cur] is not None:
                op[cur].wait()

            # Split into in projection weight andn out projection weight
            sub_qkv_proj = sub_w1_list[cur]
            # (hidden_size/p, hidden_size)
            sub_out_proj = sub_w2_list[cur]

            # Prepare sub qkv for Inner Loop
            # (b, seq_len/p, hidden_size) @ (hidden_size, hidden_size/p) -> (b, seq_len/p, 3*hidden_size/p)
            sub_qkv = torch.matmul(sub_hidden_states, sub_qkv_proj.t())

            # Inner Loop Attention Computation ...

            # Output Projections backward
            # attn_output shape: [batch_size, sub_seq_len, sub_num_heads, head_size]
            out_project_input = attn_output_list[i].reshape(-1, sub_hidden_size)

            d_sub_out_proj = torch.matmul(out_project_input.t(), grad_output)  # [h/p, B/p] @ [B/p, h] -> [h/p, h]
            sub_delta_attn = torch.matmul(grad_output, sub_out_proj.t())  # [B/p, h] @ [h, h/p] -> [B/p, h/p]

            sub_delta_attn = sub_delta_attn.reshape(batch_size, sub_seq_length,
                                                    num_attention_heads // local_world_size,
                                                    head_size)
            # (b, seq_len/p, head_num/p, 3 * head_size) -> (3, b, seq_len/p, head_num/p , head_size)
            sub_qkv = sub_qkv.view(batch_size, sub_seq_length, sub_head_num, 3 * head_size)

            sub_q, sub_k, sub_v = torch.chunk(sub_qkv, 3, dim=-1)
            # sub_kv: [2, batch_size, num_heads, seq_len/p, head_size]
            sub_kv = torch.stack([sub_k, sub_v], dim=0)

            sub_d_q, sub_d_kv = ring_attention_backward(sub_delta_attn, sub_q, sub_kv, L_list[i], attn_output_list[i],
                                                        rng_list_list[i],
                                                        no_comm=config.no_comm,
                                                        async_op=config.async_op,
                                                        skip_comm=config.skip_comm)

            sub_dk, sub_dv = sub_d_kv

            # sub_d_qkv: [batch_size * num_heads, sub_seq_len, 3*head_size]
            sub_d_qkv = torch.cat([sub_d_q, sub_dk, sub_dv], dim=-1).reshape(-1, 3 * sub_hidden_size)

            # for d_input, do accumulation
            # delta X = delta Y @ W^T. # [B/p, 3h/p] @ [3h/p, h]->[B/p, h]
            d_input += torch.matmul(sub_d_qkv, sub_qkv_proj)

            # for sub_in_proj, do reduction over group
            # delta W^T = delta Y^T @ X. # [3h/p, B/p] @ [B/p, h] -> [3h/p, h]
            d_sub_in_proj = torch.matmul(sub_d_qkv.t(),
                                         sub_hidden_states.reshape(-1, hidden_size))  # [3h/p, B/p] @ [B/p, h]

            d_sub_w = torch.cat([d_sub_in_proj.flatten(), d_sub_out_proj.flatten()]).contiguous()

            if reduce_op_list[cur] is not None:
                reduce_op_list[cur].wait()
                if rank == i - 2:
                    # Assign result for rank i-2

                    d_sub_w_result.copy_(reduce_buffer_list[cur])

            reduce_buffer_list[cur].copy_(d_sub_w)
            reduce_op_list[cur] = dist.reduce(reduce_buffer_list[cur], dst=i,
                                              async_op=config.async_op) if not config.skip_comm else None
            cur = 1 - cur
        for op in reduce_op_list:
            if op is not None:
                op.wait()

        # Assign d_sub_w for last two ranks.
        if rank == local_world_size - 2:
            d_sub_w_result = reduce_buffer_list[0]
        elif rank == local_world_size - 1:
            d_sub_w_result = reduce_buffer_list[1]

        d_sub_w = d_sub_w_result.reshape(4 * sub_hidden_size, hidden_size).contiguous()

        d_input = d_input.reshape(batch_size, sub_seq_length, hidden_size)
        return d_input, d_sub_w, None, None, None, None, None, None


class METPFFNFunc(torch.autograd.Function):
    """
    Calculate FFN in a flash-attention and ring-exchange style.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, sub_x, sub_w1w2, batch_size, hidden_size, sub_seq_length, config):
        """
        sub_x: [N, hidden_size], N could be multi-dimensional
        sub_w1w2: [2*hidden_size, intermediate_size / dp_degree]
        """

        # save tensor for backward
        ctx.save_for_backward(sub_x, sub_w1w2)
        ctx.dropout_p = config.dropout
        ctx.hidden_size = sub_x.shape[-1]
        ctx.sub_intermediate_size = sub_w1w2.shape[-1]
        ctx.no_comm = config.no_comm
        ctx.async_op = config.async_op

        if config.no_comm:
            config.skip_comm = True
        ctx.skip_comm = config.skip_comm

        dropout_mask_list = []
        local_world_size = dist.get_world_size() if not config.no_comm else 1
        ctx.local_world_size = local_world_size

        # ctx.group = group
        output = torch.zeros_like(sub_x, dtype=sub_x.dtype, device=sub_x.device)

        sub_w1w2_list = [get_global_memory_buffer().get_tensor(sub_w1w2.shape,
                                                               sub_w1w2.dtype,
                                                               f"metp_ffn_{i}") for i in range(2)]
        sub_w1w2_list[0] = sub_w1w2

        op = [None, None]
        cur = 0

        # compute attention in ring-all-reduce style
        for i in range(local_world_size):
            # Overlap outer-loop communication with inner-loop computation
            if op[cur] is not None:
                for req in op[cur]:
                    req.wait()

            if i < local_world_size - 1:
                # sub_w1w2_list[1 - cur] = torch.empty_like(sub_w1w2_list[cur])
                op[1 - cur] = async_ring_forward(sub_w1w2_list[cur],
                                                 sub_w1w2_list[1 - cur]) if not config.skip_comm else None
                if not config.async_op and op[1 - cur] is not None:
                    for req in op[1 - cur]:
                        req.wait()

            sub_w1, sub_w2 = torch.chunk(sub_w1w2_list[cur], 2, dim=0)
            cur = 1 - cur

            # Compute
            sub_w2 = sub_w2.t()
            sub_yi = torch.matmul(sub_x, sub_w1)
            sub_yi = torch.nn.functional.gelu(sub_yi)  # [N, intermediate_size / dp_degree]
            if 1 > config.dropout > 0:
                sub_yi, mask = dropout_forward(sub_yi, config.dropout)
                dropout_mask_list.append(mask)
            output += torch.matmul(sub_yi, sub_w2)  # [N, hidden_size]

        if 1 > config.dropout > 0:
            ctx.dropout_mask_list = dropout_mask_list  # [N, intermediate_size] * 1Bytes
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """

        :param ctx:
        :param grad_output: [batch_size, seq_len/p, hidden_size]
        :return:
        """
        # Reshape to [batch_size*seq_len/p, hidden_size]
        sub_x, sub_w1w2 = ctx.saved_tensors
        sub_x_shape = sub_x.shape
        grad_output = grad_output.reshape(-1, ctx.hidden_size)
        sub_x = sub_x.reshape(-1, ctx.hidden_size)

        local_world_size = ctx.local_world_size
        dropout_p = ctx.dropout_p
        skip_comm = ctx.skip_comm
        async_op = ctx.async_op
        if 1 > dropout_p > 0:
            dropout_mask_list = ctx.dropout_mask_list
        sub_w1w2_dw1w2 = torch.zeros((2, *sub_w1w2.shape), dtype=sub_x.dtype, device=sub_x.device)

        sub_w1w2_dw1w2[0] = sub_w1w2
        flatten_sub_w1w2_dw1w2 = sub_w1w2_dw1w2.flatten()

        dx = torch.zeros_like(sub_x, dtype=sub_x.dtype, device=sub_x.device)
        buffer_list = [torch.zeros_like(flatten_sub_w1w2_dw1w2) for _ in range(2)]
        buffer_list[0] = flatten_sub_w1w2_dw1w2

        w1w2_numel = sub_w1w2.numel()
        sub_w1_list = [buffer_list[i][:w1w2_numel].reshape(sub_w1w2.shape).chunk(2, dim=0)[0] for i in range(2)]
        sub_w2_list = [buffer_list[i][:w1w2_numel].reshape(sub_w1w2.shape).chunk(2, dim=0)[1] for i in range(2)]
        sub_dw1_list = [buffer_list[i][w1w2_numel:2 * w1w2_numel].reshape(sub_w1w2.shape).chunk(2, dim=0)[0] for i in
                        range(2)]
        sub_dw2_list = [buffer_list[i][w1w2_numel:2 * w1w2_numel].reshape(sub_w1w2.shape).chunk(2, dim=0)[1] for i in
                        range(2)]

        op = [None, None]
        cur = 0
        # compute attention in ring-all-reduce style
        for i in range(local_world_size):
            # Overlap outer-loop communication with inner-loop computation
            if i == 0:
                # Send w1w2 only
                op[1 - cur] = async_ring_forward(buffer_list[cur][:w1w2_numel],
                                                 buffer_list[1 - cur][:w1w2_numel]) if not skip_comm else None
            elif i == local_world_size - 1:
                # Send dw1w2 only
                op[1 - cur] = async_ring_forward(buffer_list[cur][w1w2_numel:],
                                                 buffer_list[1 - cur][w1w2_numel:]) if not skip_comm else None
            else:
                # Send w1w2 and dw1w2
                op[1 - cur] = async_ring_forward(buffer_list[cur], buffer_list[1 - cur]) if not skip_comm else None
            if op[1 - cur] is not None and not async_op:
                for req in op[1 - cur]:
                    req.wait()
            # Recompute
            temp = torch.matmul(sub_x, sub_w1_list[cur])
            sub_yi = torch.nn.functional.gelu(temp)  # [N, INTERMEDIATE_SIZE / dp_degree]
            if 1 > dropout_p > 0:
                sub_yi, _ = dropout_forward(sub_yi, mask=dropout_mask_list[i])
            sub_dw2_ = torch.matmul(grad_output.t(), sub_yi)  # [INTERMEDIATE_SIZE / dp_degree, N] @ [N, HIDDEN_SIZE]
            # Note: W2 is stored in a transpose form.
            d_sub_yi = torch.matmul(grad_output, sub_w2_list[cur])  # [N, INTERMEDIATE_SIZE / dp_degree]
            if 1 > dropout_p > 0:
                d_sub_yi = dropout_backward(d_sub_yi, dropout_mask_list[i], 0.1)
            d_temp = gelu_backward(d_sub_yi, temp)

            sub_dw1_ = torch.matmul(sub_x.t(), d_temp)
            dx += torch.matmul(d_temp, sub_w1_list[cur].t())

            if op[1 - cur] is not None:
                for req in op[1 - cur]:
                    req.wait()
            sub_dw1 = sub_dw1_list[1 - cur]
            sub_dw2 = sub_dw2_list[1 - cur]
            sub_dw1 += sub_dw1_
            sub_dw2 += sub_dw2_

            cur = 1 - cur

        ops = async_ring_forward(buffer_list[cur][1], buffer_list[1 - cur][1]) if not skip_comm else None
        if ops is not None:
            for op in ops:
                op.wait()
        dw1w2 = buffer_list[1 - cur][w1w2_numel:2 * w1w2_numel].reshape(2 * ctx.hidden_size, ctx.sub_intermediate_size)
        dx = dx.reshape(sub_x_shape)
        return dx, dw1w2, None, None, None, None
