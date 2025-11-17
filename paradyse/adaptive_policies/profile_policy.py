from .policy_base import PolicyBase
from ..configs import TransformerConfig
import torch.distributed as dist
import GPUtil

DATA_TYPE_FP16 = 2
MB = 1024 * 1024


class ProfilePolicy(PolicyBase):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.name = "Profile Policy"
        self.profile_dict = {}
        self.config = config

        self.model_parameters_num = self.config.hidden_size ** 2 * 12 * self.config.num_layers
        self.model_states_memory = self.model_parameters_num * DATA_TYPE_FP16 * 4  # 2 for fp16 and 4 for parameters, gradients and optimizer states

        self.world_size = dist.get_world_size()

        gpus = GPUtil.getGPUs()
        self.memory_limit = gpus[0].memoryTotal * MB

    def get_megatron_extra_memory(self, batch_size, seq_len):
        bsh = batch_size * seq_len * self.config.hidden_size * DATA_TYPE_FP16

        # 10 = 3 (qkv) + 1 (attn_output) + 1 (attn_proj) + 4 (ffn in_proj) + 1 (ffn out_proj)
        static_activation_memory_per_layer = 10 * bsh / self.world_size

        # Peak： 有一个AG后的张量，或者RS前的张量的开销为bsh。
        peak_memory_offset = bsh - bsh / self.world_size

        megatron_extra_memory = static_activation_memory_per_layer * self.config.num_layers + peak_memory_offset
        return megatron_extra_memory

    def get_ulysses_extra_memory(self, batch_size, seq_len):
        bsh = batch_size * seq_len * self.config.hidden_size * DATA_TYPE_FP16

        static_activation_memory_per_layer = 10 * bsh / self.world_size

        # Peak：假设每次对一个linear层AllGather完后就释放，则Peak会在ffn in_proj处产生。
        peak_memory_offset = (4 * self.config.hidden_size ** 2 * DATA_TYPE_FP16) * (1 - 1 / self.world_size)

        ulysses_extra_memory = static_activation_memory_per_layer * self.config.num_layers + peak_memory_offset
        return ulysses_extra_memory

    def get_cp_memory(self, batch_size, seq_len):
        """ 注意： 本仓库的实现中，CP和ulysses的内存行为相似"""
        bsh = batch_size * seq_len * self.config.hidden_size * DATA_TYPE_FP16

        static_activation_memory_per_layer = 10 * bsh / self.world_size

        # Peak：假设每次对一个linear层AllGather完后就释放，则Peak会在ffn in_proj处产生。
        zero_peak_memory_offset = (4 * self.config.hidden_size ** 2 * DATA_TYPE_FP16) * (1 - 1 / self.world_size)

        cp_extra_memory = static_activation_memory_per_layer * self.config.num_layers + zero_peak_memory_offset

        return cp_extra_memory

    def get_metp_memory(self, batch_size, seq_len):
        bsh = batch_size * seq_len * self.config.hidden_size * DATA_TYPE_FP16

        static_activation_memory_per_layer = 2 * bsh / self.world_size  # 2 for fp 16

        # Peak：FFN的计算过程中产生一个4bsh/p^2大小的张量。
        peak_memory_offset = (4 * bsh * DATA_TYPE_FP16) / (self.world_size ** 2)

        metp_extra_memory = static_activation_memory_per_layer * self.config.num_layers + peak_memory_offset
        return metp_extra_memory

    def select_method(self, batch_size, seq_len):
        if (batch_size, seq_len) not in self.profile_dict:
            self.profile_dict[(batch_size, seq_len)] = dict()

        megatron_memory = self.get_megatron_extra_memory(batch_size, seq_len)
        ulysses_memory = self.get_ulysses_extra_memory(batch_size, seq_len)
        cp_memory = self.get_cp_memory(batch_size, seq_len)
        metp_memory = self.get_metp_memory(batch_size, seq_len)

        options_memory_dict = {"megatron": megatron_memory, "ulysses": ulysses_memory, "context_parallel": cp_memory,
                               "metp": metp_memory}
        legal_options = [k for k, v in options_memory_dict.items() if
                         v < self.memory_limit]
        if len(legal_options) == 0:
            raise ValueError(f"No legal options for this batch size {batch_size} and sequence length {seq_len}")

        non_profiled_legal_options = [k for k in legal_options if k not in self.profile_dict[(batch_size, seq_len)]]

        if len(non_profiled_legal_options) > 0:
            # 按默认顺序选择第一个合法的选项
            return non_profiled_legal_options[0]
        else:
            # 取所有合法选项中profile记录的时间最短的选项
            # print(self.profile_dict[(batch_size, seq_len)])
            return min(self.profile_dict[(batch_size, seq_len)], key=self.profile_dict[(batch_size, seq_len)].get)

    def update_profile(self, batch_size, seq_len, method, time):
        if (batch_size, seq_len) not in self.profile_dict:
            self.profile_dict[(batch_size, seq_len)] = dict()
        if method in self.profile_dict[(batch_size, seq_len)]:
            recorded_time = self.profile_dict[(batch_size, seq_len)][method]
            # 平滑时间，以最新时间为准，避免 sudden spike
            new_time = recorded_time * 0.1 + time * 0.9
        else:
            new_time = time
        self.profile_dict[(batch_size, seq_len)][method] = new_time
