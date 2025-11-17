from paradyse.adaptive_policies.policy_base import PolicyBase
from typing import List
from paradyse.configs import TransformerConfig
import torch.distributed as dist
from paradyse.utils import dna_predicted_model, github_code_predicted_model, dna_polynomial_predicted_model

MB = 1024 * 1024

class MemoryCalculator:
    """
    内存计算器类，用于计算不同并行策略下的内存开销和运行时间
    新增：支持exclude_strategy参数，排除单一策略
    """
    def __init__(self, num_heads, num_layers, seq_len, hidden_size, dataset, exclude_strategy=None):
        # exclude_strategy: 可选，排除某个策略名（如 'megatron'），为None时全部初始化
        all_methods = []
        self.oom_dict = {}
        
        if dataset == "dna":
            predicted_model = dna_predicted_model
        elif dataset == "github_code":
            predicted_model = github_code_predicted_model
        elif dataset == "dna_polynomial":
            predicted_model = dna_polynomial_predicted_model
        else:
            raise ValueError("Unsupported dataset. Use 'dna' or 'github_code'.")
        
        # 只初始化未被排除的策略
        if exclude_strategy != 'megatron':
            megatron_time_cost, megatron_mem_cost, megatron_oom, model_type = predicted_model.predict(
                L=num_layers, H=hidden_size, n=num_heads, strategy='megatron', seq_length=seq_len)
            self.megatron = ["megatron", megatron_mem_cost / num_layers, megatron_time_cost / num_layers]
            self.oom_dict['megatron'] = megatron_oom
            all_methods.append(self.megatron)
        if exclude_strategy != 'ulysses':
            ulysses_time_cost, ulysses_mem_cost, ulysses_oom, model_type = predicted_model.predict(
                L=num_layers, H=hidden_size, n=num_heads, strategy='ulysses', seq_length=seq_len)
            self.ulysses = ["ulysses", ulysses_mem_cost / num_layers, ulysses_time_cost / num_layers]
            self.oom_dict['ulysses'] = ulysses_oom
            all_methods.append(self.ulysses)
        if exclude_strategy != 'context_parallel':
            cp_time_cost, cp_mem_cost, cp_oom, model_type = predicted_model.predict(
                L=num_layers, H=hidden_size, n=num_heads, strategy='context_parallel', seq_length=seq_len)
            self.context_parallel = ["context_parallel", cp_mem_cost / num_layers, cp_time_cost / num_layers]
            self.oom_dict['context_parallel'] = cp_oom
            all_methods.append(self.context_parallel)
        if exclude_strategy != 'metp':
            metp_time_cost, metp_mem_cost, metp_oom, model_type = predicted_model.predict(
                L=num_layers, H=hidden_size, n=num_heads, strategy='metp', seq_length=seq_len)
            self.metp = ["metp", metp_mem_cost / num_layers, metp_time_cost / num_layers]
            self.oom_dict['metp'] = metp_oom
            all_methods.append(self.metp)
        # 按照时间开销、内存开销从小到大的策略排序
        self.methods_sorted = sorted(all_methods, key=lambda x: (x[2], x[1]))
        # 移除不必要的策略
        self.pop_useless()

    def pop_useless(self):
        """
        移除不必要的策略
        """
        methods_num = len(self.methods_sorted)
        if methods_num == 1:
            pass
        elif methods_num == 2:
            if self.methods_sorted[1][1] >= self.methods_sorted[0][1]:
                self.methods_sorted.pop(1)
        # 如果有多种方法，逐个比较
        # 如果当前方法的内存开销小于下一个方法，则移除下一个方法, 直到没有更多可以移除的方法
        else:
            i = 0
            while i < methods_num - 2:
                if self.methods_sorted[i+1][1] >= self.methods_sorted[i][1]:
                    self.methods_sorted.pop(i+1)
                    methods_num -= 1
                else:
                    i += 1

    def select_min_oom(self, strategy_list):
        """
        根据策略列表筛选OOM值并返回最小值
        """
        
        # 筛选出策略列表中对应的OOM值
        selected_oom_values = []
        
        for strategy in strategy_list:
            # 构造OOM键名
            if strategy in self.oom_dict:
                selected_oom_values.append(self.oom_dict[strategy])
            else:
                print(f"警告: 策略 '{strategy}' 对应的OOM值 '{strategy}' 在字典中未找到")
        
        # 如果没有找到任何匹配的OOM值，返回None
        if not selected_oom_values:
            print("错误: 没有找到任何匹配的OOM值")
            return 0
        
        # 返回最小值
        return min(selected_oom_values)

class HeuristicPolicy(PolicyBase):
    def __init__(self, config: TransformerConfig):
        """
        初始化启发式策略
        
        参数:
            config: 包含策略配置参数的字典（可选）
        """
        super().__init__()
        self.name = "Heuristic Policy"
        self.profile_dict = {} #(batch_size, seq_length) -> {method list} time
        self.config = config
        self.world_size = dist.get_world_size()
        self.memory_limit = self.memory_predicted = 0
        self.oom_flag = False

    def get_current_memory(self, method_list: list) -> float:
        """
        用于获取给定的一组并行策略的内存占用大小。
        """
        memory = 0
        for method in method_list:
            method_obj = getattr(self.memlator, method)
            memory += method_obj[1]  # 累加 memory_per_layer
        return memory
    
    def get_list_time(self, method_list: list) -> float:
        """
        用于获取给定的一组并行策略的运行时间。
        """
        time = 0
        for method in method_list:
            method_obj = getattr(self.memlator, method)
            time += method_obj[2]
        return time

    def compare_memory(self, method_list: list) -> bool: # , exchange_position: int = 0
        """
        用于比较给定的一组并行策略是否超限；如果并行策略有两种，要判断切换策略时是否超限。exchange_position为0表示没有改变并行策略，为其他任意正整数n表示在第n层之后发生了策略切换。
        """
        current_memory = self.get_current_memory(method_list)
        self.memory_limit = self.memlator.select_min_oom(method_list) #- 2 * 1024  # OOM阈值
        if current_memory < self.memory_limit:
            return True
        else:
            return False

    def compare_prior_time_cost(self, method_list: list, prior_method_list: list) -> list:
        """
        用于比较给定的前一个序列的策略在当前序列下的时间开销是否大于启发式算法选择的策略的时间开销的5%。
        """
        current_list_time = self.get_list_time(method_list)
        prior_list_time = self.get_list_time(prior_method_list)
        if prior_list_time < (current_list_time * 1.05):
            if self.compare_memory(prior_method_list):
                return prior_method_list
            else:
                return method_list
        else:
            return method_list

        
    def flush_memory_limit(self, selected_methods):
        """
        刷新内存限制
        """
        self.memory_predicted = self.get_current_memory(selected_methods)
        self.memory_limit = self.memlator.select_min_oom(selected_methods) - 512 # OOM阈值

    def select_method(self, batch_size: int, seq_len: int, prior_seq_length: int, dataset: str, exclude_strategy = None, smoothing = True) -> List[str] | None:
        """
        为所有层生成并行策略列表
        
        参数:
            batch_size: 批大小
            seq_len: 序列长度
            **kwargs: 扩展参数（用于未来扩展）
            
        返回:
            每层的策略列表
        """
        self.memlator = MemoryCalculator(
            num_heads=self.config.num_attention_heads,
            num_layers=self.config.num_layers,
            seq_len=seq_len,
            hidden_size=self.config.hidden_size,
            exclude_strategy=exclude_strategy,
            dataset=dataset
        )
        if not smoothing:
            prior_seq_length = 0
        if (batch_size, seq_len) not in self.profile_dict: # seq_length不在记录的字典中
            self.profile_dict[(batch_size, seq_len)] = dict()
            if prior_seq_length != 0:
                prior_selected_methods = list(list(self.profile_dict[(batch_size, prior_seq_length)].keys())[0])

            if self.oom_flag: # 如果上一个序列已经判断超出OOM阈值，下一个序列必定超出阈值，直接返回metp
                selected_methods = ["metp"] * self.config.num_layers
                self.flush_memory_limit(selected_methods)
                return selected_methods
            i = 0
            method_options = []
            method_num = len(self.memlator.methods_sorted)
            while i < method_num:
                selected_methods: list[str] = [self.memlator.methods_sorted[i][0]] * self.config.num_layers # 看不需要切换策略时能否满足要求
                if self.compare_memory(selected_methods) and i == 0: # 最快的方法内存都没超限，直接返回最快策略
                    if prior_seq_length != 0:
                        if prior_selected_methods != selected_methods:
                            selected_methods = self.compare_prior_time_cost(selected_methods, prior_selected_methods)
                    self.flush_memory_limit(selected_methods)
                    return selected_methods
                elif self.compare_memory(selected_methods): # 如果不是最快的方法，先加到待选列表中
                    method_options.append([selected_methods, self.get_list_time(selected_methods)])
                else: # 不是最快的方法，且显存占用大于已知OOM阈值，判断与其后面策略组合起来是否能小于OOM阈值
                    k = i + 1 # 记录当前正在遍历的i策略后面策略的指针
                    while k < method_num:
                        selected_methods_tmp: list = [self.memlator.methods_sorted[k][0]] * self.config.num_layers
                        if self.compare_memory(selected_methods_tmp): # 判断下一级并行策略的内存占用是否有切换价值（是否足够小）
                            exchange_position = 1
                            while exchange_position < self.config.num_layers: # 从第1个位置开始切换策略，逐个位置遍历
                                selected_methods_tmp.pop(-1)
                                selected_methods_tmp.insert(0, self.memlator.methods_sorted[i][0]) # 将当前i方法插入到k策略列表第一个位置
                                if self.compare_memory(selected_methods_tmp): # 策略小于当前限制
                                    method_options.append([selected_methods_tmp, self.get_list_time(selected_methods_tmp)])
                                else: # 后续没有判断价值了，直接跳出循环
                                    exchange_position = self.config.num_layers
                                exchange_position += 1
                            k += 1
                        else: # 切换下一级策略
                            k += 1
                i += 1 # 继续判断下一种方法
            if method_options: # 待选列表中有很多个策略，选出其中时间最短的
                min_methods_options = min(method_options, key=lambda x: x[1])[0]
                if prior_seq_length != 0:
                    if prior_selected_methods != min_methods_options:
                        min_methods_options = self.compare_prior_time_cost(min_methods_options, prior_selected_methods)
                self.flush_memory_limit(min_methods_options)
                return min_methods_options
            else: # 待选列表为空，表示我们判断所有方法都会OOM，返回metp
                self.oom_flag = True
                selected_methods = ["metp"] * self.config.num_layers
                self.flush_memory_limit(selected_methods)
                return selected_methods  # 如果没有合适的策略，返回默认的metp策略
        else: # 若记录字典中有seq_length记录，直接使用记录策略列表
            selected_methods = list(list(self.profile_dict[(batch_size, seq_len)].keys())[0])
            self.flush_memory_limit(selected_methods)
            return selected_methods

    def update_profile(self, batch_size, seq_len, method, time):
        if isinstance(method,list):
            method = tuple(method)
        if method in self.profile_dict[(batch_size, seq_len)]:
            recorded_time = self.profile_dict[(batch_size, seq_len)][method]
            # 平滑时间，以最新时间为准，避免 sudden spike
            new_time = recorded_time * 0.1 + time * 0.9
        else:
            new_time = time
        self.profile_dict[(batch_size, seq_len)][method] = new_time

if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=10000, 
        num_layers=5, 
        hidden_size=8192,
        num_heads=12
    )
    her = HeuristicPolicy(config)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int)
    args = parser.parse_args()
    seq_length = args.s
    # policy = her.select_method(1,seq_length)
    # print(policy)