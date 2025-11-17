import torch
from torch import distributed as dist
from torch.amp import custom_fwd, custom_bwd

from paradyse.nn.functional import torch_attention
from paradyse.utils.global_buffer import get_global_memory_buffer


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = dist.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
            dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous()
    )
    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = dist.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous()
    )

    return output


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
            ctx,
            input: torch.Tensor,
            weight: torch.Tensor,
            sequence_parallel,
    ):
        ctx.save_for_backward(input, weight)
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            world_size = dist.get_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed.all_gather_into_tensor(
                all_gather_buffer, input.contiguous()
            )
            total_input = all_gather_buffer
        else:
            total_input = input
        output = torch.matmul(total_input, weight.t())
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        if ctx.sequence_parallel:
            world_size = dist.get_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer, input, async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.reshape(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        if ctx.sequence_parallel:
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed.reduce_scatter_tensor(
                sub_grad_input, grad_input, async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        grad_weight = grad_output.t().matmul(total_input)

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, None

        return grad_input, grad_weight, None


def megatron_mha_function(sub_hidden_states, sub_projection_w, batch_size, hidden_size, sub_seq_length,
                          num_attention_heads, config):
    """Tensor parallel + Sequence Parallel Multi-Head Self-Attention Function

    Args:
        sub_hidden_states:  [batch_size, sub_seq_len/p, hidden_size] # seq_length first layout.
        sub_projection_w: [4*hidden_size/p, hidden_size], concat of W1^T and W2
        batch_size: batch_size
        num_attention_heads: num_heads. head_size = hidden_size / num_attention_heads.
        sub_seq_length: sub_seq_len = sequence_length / parallel_size
        hidden_size: hidden_size
    Returns:
        output : [sub_seq_len/p, batch_size, hidden_size]

    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    w1_t = sub_projection_w[:3 * hidden_size // world_size]
    w2 = sub_projection_w[3 * hidden_size // world_size:]

    # hidden_states: [sub_seq_len, batch_size, hidden_size]
    batch_size, sub_seq_length, hidden_size = sub_hidden_states.size()

    # =====================
    # Query, Key, and Value
    # =====================

    # Permuting the input tensor to have seq_len first to do allgather
    sub_hidden_states = sub_hidden_states.permute(1, 0, 2).contiguous()
    # Attention heads shape change:
    # [sub_seq_len, batch_size, hidden_size] --> [seq_len, batch_size, (num_heads/p * 3*head_size)]
    mixed_x_layer = LinearWithGradAccumulationAndAsyncCommunication.apply(sub_hidden_states,
                                                                          w1_t,
                                                                          True)

    # [seq_len, batch_size, (3 * num_heads/p * head_size)] --> [seq_len, batch_size, num_heads/p, 3*head_size)]
    hidden_size_per_attention_head = hidden_size // num_attention_heads
    new_tensor_shape = mixed_x_layer.size()[:-1] + (num_attention_heads // world_size,
                                                    3 * hidden_size_per_attention_head)
    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

    # split into query, key and value
    last_dim_value = mixed_x_layer.size(-1)
    assert last_dim_value % 3 == 0, 'the last dimension is not a multiple of 3, ' \
                                    'cannot be divided into query, key and value'

    mixed_x_layer = mixed_x_layer.permute(1, 0, 2, 3)
    # [batch_size, seq_len, num_heads, 3*head_size]
    output = torch_attention(mixed_x_layer)

    # permute to have seq_len first to do reduce_scatter
    # [b, seq_len, hp/p] -> [b, seq_len, hidden_size]
    # output = self.dense(output)
    output = LinearWithGradAccumulationAndAsyncCommunication.apply(output,
                                                                   w2.t(),
                                                                   False)
    output = output.permute(1, 0, 2)
    # [sub_seq_len, b, hidden_size]
    output = reduce_scatter_to_sequence_parallel_region(output)
    output = output.permute(1, 0, 2)

    return output


def megatron_ffn_function(hidden_states, sub_projection_w, batch_size, hidden_size, sub_seq_length, config):
    """Tensor parallel + Sequence Parallel Feedforward function
     在FFN中不需要沿着seq_len维度做通信，因此不需要permute.

    Args:
        hidden_states:  [batch_size, sub_seq_len/p,, hidden_size] # batch_size first layout.
        sub_projection_w: [2*hidden_size, intermediate_size / dp_degree] concat of W1 and W2^T
        batch_size: batch_size
        sub_seq_length: sub_seq_len = sequence_length / parallel_size
        hidden_size: hidden_size
    Returns:
        output : [batch_size, sub_seq_len/p,, hidden_size]

    """
    w1 = sub_projection_w[:hidden_size]
    w2_t = sub_projection_w[hidden_size:]

    # hidden_states: [sub_seq_len, batch_size, hidden_size]
    sub_seq_length, batch_size, hidden_size = hidden_states.size()

    # column_tp_output_parallel shape: [batch_size*p, sub_seq_len/p, hidden_size]
    # 在FFN中不需要沿着seq_len维度做通信。
    column_tp_output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(hidden_states,
                                                                                      w1.t(),
                                                                                      True)
    gelu_output_parallel = torch.nn.functional.gelu(column_tp_output_parallel)

    output = LinearWithGradAccumulationAndAsyncCommunication.apply(gelu_output_parallel,
                                                                   w2_t,
                                                                   False)
    output = reduce_scatter_to_sequence_parallel_region(output)

    return output
