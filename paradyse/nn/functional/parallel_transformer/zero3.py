from typing import Union

import torch
import torch.distributed as dist
from torch.amp import custom_fwd, custom_bwd
from torch.distributed import all_gather_into_tensor
from paradyse.communication.async_ring import async_ring_forward
from paradyse.nn.functional import torch_attention
from paradyse.utils.global_buffer import get_global_memory_buffer


class ColumnWiseRingZeRO3LinearFunc(torch.autograd.Function):
    """
    Calculate ZeRO3 Linear in a ring style. weight is column-wise split.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x: torch.Tensor, sub_weight: torch.Tensor, group=None):
        """
        :args
            x: [N / dp_degree, in_size]
            sub_weight: [in_size, out_size / dp_degree]
        :returns
            output: [N / dp_degree, out_size]
        """

        # save tensor for backward

        local_world_size = dist.get_world_size()
        ctx.local_world_size = local_world_size
        output_list: list[Union[None, torch.Tensor]] = [None] * local_world_size
        local_rank = dist.get_rank()

        sub_w_list = [get_global_memory_buffer().get_tensor(sub_weight.shape,
                                                            sub_weight.dtype,
                                                            f"zero3_{i}") for i in range(2)]
        # 注意send_recv所输入的张量需要在内存空间上连续
        sub_w_list[0] = sub_weight.contiguous()
        op = [None, None]
        cur = 0

        # compute attention in ring-all-reduce style
        for i in range(local_world_size):
            # Overlap outer-loop communication with inner-loop computation
            if op[cur] is not None:
                for req in op[cur]:
                    req.wait()

            if i < local_world_size - 1:
                # sub_w_list[1 - cur] = torch.empty_like(sub_w_list[cur])
                op[1 - cur] = async_ring_forward(sub_w_list[cur], sub_w_list[1 - cur])

            temp_weight = sub_w_list[cur]
            cur = 1 - cur

            sub_output = torch.matmul(x, temp_weight)
            # output_list.append(sub_output)
            output_list[(local_rank - i) % local_world_size] = sub_output

        output = torch.cat(output_list, dim=-1)
        ctx.save_for_backward(x, sub_weight)

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, sub_weight = ctx.saved_tensors
        local_world_size = ctx.local_world_size
        local_rank = dist.get_rank()
        output_size = grad_output.size(-1)
        input_size = x.size(-1)

        dx = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        sub_w_dw = torch.zeros((2, *sub_weight.shape), dtype=sub_weight.dtype, device=x.device)
        sub_w_dw[0].copy_(sub_weight)

        # 这里需要用zero初始化，因为sub_dw是要累加的。由于sub_dw要return，所以不用global_buffer
        sub_w_dw_list = [sub_w_dw, torch.zeros_like(sub_w_dw)]
        # op = [None] * 2
        op = [None, None]
        cur = 0
        grad_output_list = torch.chunk(grad_output, local_world_size, dim=-1)

        # compute attention in ring-all-reduce style
        for i in range(local_world_size):
            index = (local_rank - i) % local_world_size
            # Overlap outer-loop communication with inner-loop computation

            # if op[cur] is not None:
            #     for req in op[cur]:
            #         req.wait()
            if i == 0:
                # Send kv only
                op[1 - cur] = async_ring_forward(sub_w_dw_list[cur][0], sub_w_dw_list[1 - cur][0])
            elif i == local_world_size - 1:
                # Send dkv only
                op[1 - cur] = async_ring_forward(sub_w_dw_list[cur][1], sub_w_dw_list[1 - cur][1])
            else:
                # Send k_i v_i and dk_{i-1} dv_{i-1}
                op[1 - cur] = async_ring_forward(sub_w_dw_list[cur], sub_w_dw_list[1 - cur])
            sub_weight = sub_w_dw_list[cur][0]

            dx += torch.matmul(grad_output_list[index], sub_weight.t())

            if op[1 - cur] is not None:
                for req in op[1 - cur]:
                    req.wait()
            sub_dw = sub_w_dw_list[cur][1]
            sub_dw_ = torch.matmul(x.view(-1, input_size).t(),
                                   grad_output_list[index].reshape(-1, output_size // local_world_size))
            sub_dw += sub_dw_
            cur = 1 - cur

        ops = async_ring_forward(sub_w_dw_list[cur][1], sub_w_dw_list[1 - cur][1])
        for op in ops:
            op.wait()

        sub_dw = sub_w_dw_list[1 - cur][1]
        return dx, sub_dw, None


class RowWiseAGZeRO3LinearFunc(torch.autograd.Function):
    """
    Calculate ZeRO3 Linear with all gather and reduce scatter.
    weight is row-wise split.
    TODO: design a ring row-wise zero linear function.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x: torch.Tensor, sub_weight: torch.Tensor, group=None):
        """
        x: [N / dp_degree, in_size]
        sub_weight: [in_size / dp_degree, out_size]
        """

        # save tensor for backward
        local_world_size = dist.get_world_size()
        ctx.local_world_size = local_world_size

        dim_size = list(sub_weight.size())
        dim_size[0] = dim_size[0] * local_world_size

        # 可以和mpu共用buffer
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, sub_weight.dtype, "mpu")
        # torch.distributed._all_gather_base(
        #     all_gather_buffer, sub_weight
        # )
        all_gather_into_tensor(all_gather_buffer, sub_weight)
        total_weight = all_gather_buffer

        output = torch.matmul(x, total_weight)

        ctx.save_for_backward(x, sub_weight)

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, sub_weight = ctx.saved_tensors
        local_world_size = ctx.local_world_size

        dim_size = list(sub_weight.size())
        dim_size[0] = dim_size[0] * local_world_size

        # 可以和mpu共用buffer
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, sub_weight.dtype, "mpu")
        # ag_handle = torch.distributed._all_gather_base(
        #     all_gather_buffer, sub_weight, async_op=True
        # )
        ag_handle = all_gather_into_tensor(all_gather_buffer, sub_weight, async_op=True)
        total_weight = all_gather_buffer

        dim_size = list(sub_weight.size())

        # (input_size, N/p) @ (N/p, output_size)
        output_size = grad_output.size(-1)
        input_size = x.size(-1)
        partial_dw = torch.matmul(x.view(-1, input_size).t(), grad_output.view(-1, output_size))
        sub_dw = torch.empty(
            dim_size, dtype=sub_weight.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        # reduce_scatter
        rs_handle = torch.distributed.reduce_scatter_tensor(
            sub_dw, partial_dw, async_op=True
        )

        # 计算dx前，ag需要完成
        ag_handle.wait()
        # (N/p, output_size) @ (output_size, input_size)
        dx = torch.matmul(grad_output, total_weight.t())

        # 返回结果前，需要完成rs
        rs_handle.wait()

        return dx, sub_dw, None


def zero_mha_function(sub_hidden_states, sub_projection_w, batch_size, hidden_size, sub_seq_length,
                      num_attention_heads, config, attn_func=torch_attention):
    """ZeRO Multi-Head Self-Attention functionalized module.

    Args:
        sub_hidden_states:  [batch_size, seq_length/p, hidden_size] # batch first layout.
        sub_projection_w: [4*hidden_size/p, hidden_size], concat of W1^T and W2
        batch_size: batch_size
        num_attention_heads: num_heads. head_size = hidden_size / num_attention_heads.
        sub_seq_length: sub_seq_len = sequence_length / parallel_size
        hidden_size: hidden_size
        attn_func: options: torch_attention, Ulysses, Ring Attention. If you use torch_attention,
            seq_len axis cannot be partitioned.
    Returns:
        output : [seq_length/p, batch_size, hidden_size]

    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # w1_t: (3*hidden_size/p, hidden_size)
    w1_t = sub_projection_w[:3 * hidden_size // world_size]
    # w2: (hidden_size/p, hidden_size)
    w2 = sub_projection_w[3 * hidden_size // world_size:]

    # hidden_states: [batch_size, sub_seq_len, hidden_size]
    hidden_states = sub_hidden_states
    batch_size, sub_seq_length, hidden_size = hidden_states.size()

    # --> [batch_size, sub_seq_len, (num_heads * 3*head_size)]
    mixed_x_layer = ColumnWiseRingZeRO3LinearFunc.apply(hidden_states,
                                                        w1_t.t(),
                                                        None)

    # [batch_size, sub_seq_len, (3 * num_heads/p * head_size)] --> [batch_size, sub_seq_len, num_heads/p, 3*head_size]
    hidden_size_per_attention_head = hidden_size // num_attention_heads
    new_tensor_shape = mixed_x_layer.size()[:-1] + (num_attention_heads,
                                                    3 * hidden_size_per_attention_head)
    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

    # =====================
    # Attention Function
    # mixed_x_layer shape: [batch_size, sub_seq_len, num_heads/p, 3*head_size]
    # attn_output shape: [batch_size, sub_seq_len, hidden_size)]
    # =====================
    attn_output = attn_func(mixed_x_layer)
    # [seq_len, b, hp] -> [sub_seq_len, b, hidden_size]
    output = RowWiseAGZeRO3LinearFunc.apply(attn_output,
                                            w2,
                                            None)
    return output


def zero_ffn_function(hidden_states, sub_projection_w, batch_size, hidden_size, sub_seq_length, config):
    """ZeRO Feed Forward Network function

    Args:
        hidden_states:  [sub_seq_len/p, batch_size, hidden_size] # seq_length first layout.
        sub_projection_w: [2*hidden_size, intermediate_size / dp_degree] concat of W1 and W2^T
        batch_size: batch_size
        sub_seq_length: sub_seq_len = sequence_length / parallel_size
        hidden_size: hidden_size
    Returns:
        output : [sub_seq_len/p, batch_size, hidden_size]

    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    w1 = sub_projection_w[:hidden_size]
    w2_t = sub_projection_w[hidden_size:]

    # hidden_states: [sub_seq_len, batch_size, hidden_size]
    sub_seq_length, batch_size, hidden_size = hidden_states.size()

    column_zero_output = ColumnWiseRingZeRO3LinearFunc.apply(hidden_states,
                                                             w1,
                                                             None)
    gelu_output_parallel = torch.nn.functional.gelu(column_zero_output)

    output = RowWiseAGZeRO3LinearFunc.apply(gelu_output_parallel,
                                            w2_t.t().contiguous(),
                                            None)
    return output
