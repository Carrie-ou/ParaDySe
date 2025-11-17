#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist


def async_ring_forward(tensor_send_next: torch.Tensor, tensor_recv_prev: torch.Tensor) -> torch.Tensor:
    """Sends a tensor to the next member and receives a tensor from the previous member.
    This function returns the received tensor from the previous member.

    Args:
        tensor_send_next (:class:`torch.Tensor`): Tensor sent to next member
        tensor_recv_prev (:class:`torch.Tensor`): Tensor recv from prev member
        parallel_mode (ParallelMode): Parallel group mode used in this communication

    Returns:
        :class:`torch.Tensor`: The tensor received from the previous.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    ops = []
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()
    next_rank = (current_rank + 1) % world_size
    prev_rank = (current_rank - 1) % world_size
    # send to next rank
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor_send_next, next_rank
    )
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor_recv_prev, prev_rank
    )
    ops.append(recv_prev_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = torch.distributed.batch_isend_irecv(ops)

    return reqs
