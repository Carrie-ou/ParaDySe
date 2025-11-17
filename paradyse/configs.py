"""config class of paradyse attention"""


class TransformerConfig:
    """Transformer config class"""

    def __init__(self,
                 num_layers=2,
                 dropout=0,
                 hidden_size=1024,
                 num_heads=16,
                 vocab_size=50000,
                 no_comm=False,
                 skip_comm=False,
                 async_op=True
                 ):
        """initialize config"""
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.no_comm = no_comm
        self.skip_comm = skip_comm
        self.async_op = async_op
        self.vocab_size = vocab_size
