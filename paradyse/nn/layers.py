""" Paradyse layers """
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter

from paradyse.nn.functional import METPFFNFunc, METPMultiHeadSelfAttentionFunc, ulysses_zero_mha_func, \
    ring_attn_zero_mha_func, megatron_ffn_function, zero_ffn_function, megatron_mha_function


class ParadyseTransformerSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h] or [b, s, h]
    and returns output of the same size.

    Args:
        config: config class
        layer_number (int): number of layers.

    """

    def __init__(
            self,
            config,
            layer_number,
    ):
        super().__init__()
        self.layer_number = layer_number
        # hidden_size (int): hidden size.
        # num_attention_heads (int): number of attention heads.

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.config = config

        assert (
                self.hidden_size % self.num_attention_heads == 0
        ), "hidden size is not divisible by the number of attention heads"

        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # weight
        self.weight = Parameter(
            torch.empty(
                4 * self.hidden_size // self.world_size,
                self.hidden_size,
            )
        )

        self.query_key_value_weight = self.weight[:3 * self.hidden_size // self.world_size, :].t()
        self.dense_weight = self.weight[3 * self.hidden_size // self.world_size:, :]

        # Note: we use kaiming_uniform_ to initialize the weights, which is
        # recommended by the original transformer paper.
        # https://arxiv.org/abs/1706.03762
        # We tried to use nn.init.uniform_ initialization and result into softmax overflow issues
        # because the q and k values are too large.

        init.kaiming_uniform_(self.query_key_value_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.dense_weight, a=math.sqrt(5))

        self.method_dict = {
            "metp": METPMultiHeadSelfAttentionFunc,
            "megatron": megatron_mha_function,
            "ulysses": ulysses_zero_mha_func,
            # "USP": USPSelfAttentionFunc,
            "context_parallel": ring_attn_zero_mha_func,
        }

    def forward(self, hidden_states, method_name="megatron"):
        if method_name == "megatron":
            sub_seq_length, batch_size, hidden_size = hidden_states.size()
        else:
            batch_size, sub_seq_length, hidden_size = hidden_states.size()

        args = [hidden_states, self.weight,
                batch_size, hidden_size,
                sub_seq_length, self.num_attention_heads,
                self.config]

        func = self.method_dict[method_name]
        if method_name == "metp":
            output = func.apply(*args)
        else:
            output = func(*args)
        return output

    def __repr__(self):
        return (
            f"ParadyseTransformerSelfAttention("
            f"layer_number={self.layer_number}, hidden_size:{self.hidden_size},  "
            f"num_attention_heads={self.num_attention_heads}, "
            f"hidden_size_per_attention_head={self.hidden_size_per_attention_head}"
        )


# TODO 改完MHA后同理修改这个
class ParadyseMLP(nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, config, layer_number):
        super(ParadyseMLP, self).__init__()

        local_world_size = dist.get_world_size() if not config.no_comm else 1
        self.no_comm = config.no_comm
        self.async_op = config.async_op
        self.layer_number = layer_number
        self.dropout_p = config.dropout
        self.config = config
        # Project to 4h.
        self.w1w2 = Parameter(
            torch.empty(
                2 * config.hidden_size,
                4 * config.hidden_size // local_world_size,
            )
        )
        init.normal_(self.w1w2)

        # self.dense_h_to_4h_weight = self.w1w2[:hidden_size, :]
        # self.dense_4h_to_h_weight = self.w1w2[hidden_size:, :]
        self.method_dict = {
            "metp": METPFFNFunc,
            "megatron": megatron_ffn_function,
            "ulysses": zero_ffn_function,
            # "USP": USPSelfAttentionFunc,
            "context_parallel": zero_ffn_function,
        }

    def forward(self, hidden_states, method_name):
        # hidden states should be in the shape of [b, s, h]
        # it will be projects into [b, s, 4h],
        # passed to a gelu activation function,
        # and projected back to [b, s, h]
        if method_name == "megatron":
            sub_seq_length, batch_size, hidden_size = hidden_states.size()
        else:
            batch_size, sub_seq_length, hidden_size = hidden_states.size()

        args = (hidden_states, self.w1w2, batch_size, hidden_size, sub_seq_length, self.config)

        func = self.method_dict[method_name]
        if method_name == "metp":
            output = func.apply(*args)
        else:
            output = func(*args)

        return output

    def __repr__(self):
        return (
            f"TransformerMLPRing(hidden_size:{self.hidden_size}, intermediate_size:{self.hidden_size * 4} "
            f"dropout p={self.dropout_p}, activation function=gelu, "
            f"parallel_degree={dist.get_world_size()})"
        )
