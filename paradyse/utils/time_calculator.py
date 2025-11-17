from typing import Any
from loguru import logger

class TimeCalculator:
    def __init__(self, seq_length, num_layers, hidden_size):
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.megatron_time = self.calculate_megatron_time()
        self.ulysses_time = self.calculate_ulysses_time()
        self.context_parallel_time = self.calculate_context_parallel_time()
        self.metp_time = self.calculate_metp_time()

    def calculate_megatron_time(self):
        if self.num_layers == 1 and self.hidden_size == 12288:
            return 1.08e-10 * self.seq_length ** 2 + 6.22e-06 * self.seq_length + 0.010117
        elif self.num_layers == 5 and self.hidden_size == 8192:
            return (3.59e-10 * self.seq_length ** 2 + 0.000016 * self.seq_length + 0.017448) / self.num_layers
        elif self.num_layers == 24 and self.hidden_size == 1024:
            return (2.36e-10 * self.seq_length ** 2 + 5.22e-06 * self.seq_length + 0.028366) / self.num_layers
        elif self.num_layers == 32 and self.hidden_size == 4096:
            return (1.23e-09 * self.seq_length ** 2 + 0.000033 * self.seq_length + 0.098259) / self.num_layers
        else:
            # logger.info("-----------------------------just for testing-----------------------------")
            return 1.08e-10 * self.seq_length ** 2 + 6.22e-06 * self.seq_length + 0.010117

    def calculate_ulysses_time(self):
        if self.num_layers == 1 and self.hidden_size == 12288:
            return 1.03e-10 * self.seq_length ** 2 + 5.07e-06 * self.seq_length + 0.093456
        elif self.num_layers == 5 and self.hidden_size == 8192:
            return (3.41e-10 * self.seq_length ** 2 + 0.000012 * self.seq_length + 0.316047) / self.num_layers
        elif self.num_layers == 24 and self.hidden_size == 1024:
            return (2.41e-10 * self.seq_length ** 2 + 7.30e-07 * self.seq_length + 0.162314) / self.num_layers
        elif self.num_layers == 32 and self.hidden_size == 4096:
            return (1.05e-09 * self.seq_length ** 2 + 0.000027 * self.seq_length + 0.970848) / self.num_layers
        else:
            # logger.info("-----------------------------just for testing-----------------------------")
            return 1.03e-10 * self.seq_length ** 2 + 5.07e-06 * self.seq_length + 0.093456

    def calculate_context_parallel_time(self):
        if self.num_layers == 1 and self.hidden_size == 12288:
            return 1.03e-10 * self.seq_length ** 2 + 5.89e-06 * self.seq_length + 0.093685
        elif self.num_layers == 5 and self.hidden_size == 8192:
            return (3.53e-10 * self.seq_length ** 2 + 0.000014 * self.seq_length + 0.342677) / self.num_layers
        elif self.num_layers == 24 and self.hidden_size == 1024:
            return (2.30e-10 * self.seq_length ** 2 + 3.37e-06 * self.seq_length + 0.222728) / self.num_layers
        elif self.num_layers == 32 and self.hidden_size == 4096:
            return (1.05e-09 * self.seq_length ** 2 + 0.000027 * self.seq_length + 0.970848) / self.num_layers
        else:
            # logger.info("-----------------------------just for testing-----------------------------")
            return 1.03e-10 * self.seq_length ** 2 + 5.89e-06 * self.seq_length + 0.093685

    def calculate_metp_time(self):
        if self.num_layers == 1 and self.hidden_size == 12288:
            return 1.04e-10 * self.seq_length ** 2 + 6.81e-06 * self.seq_length + 0.112750
        elif self.num_layers == 5 and self.hidden_size == 8192:
            return (3.33e-10 * self.seq_length ** 2 + 0.000020 * self.seq_length + 0.290053) / self.num_layers
        elif self.num_layers == 24 and self.hidden_size == 1024:
            return (3.28e-10 * self.seq_length ** 2 + -0.000012 * self.seq_length + 1.368680) / self.num_layers
        elif self.num_layers == 32 and self.hidden_size == 4096:
            return (1.18e-09 * self.seq_length ** 2 + 0.000039 * self.seq_length + 1.436488) / self.num_layers
        else:
            # logger.info("-----------------------------just for testing-----------------------------")
            return 1.04e-10 * self.seq_length ** 2 + 6.81e-06 * self.seq_length + 0.112750

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
