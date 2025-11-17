"""使用Paradyse的layer接口实现的TransformerLayer

"""

import torch.nn as nn

from paradyse.nn.layers import ParadyseTransformerSelfAttention, ParadyseMLP


class TransformerLayer(nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
            self,
            layer_number,
            config
    ):
        super().__init__()
        self.layer_number = layer_number

        # Self attention.
        self.attention = ParadyseTransformerSelfAttention(
            config, layer_number=layer_number,
        )

        # FFN
        self.mlp = ParadyseMLP(config, layer_number=layer_number)

        # Layer norm on the attention output
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        # Layer norm on the FFN output
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, method_name):
        # hidden_states: [batch_size, sub_seq_len, hidden_size]

        # Layer norm at the beginning of the transformer layer.

        # Self attention.
        attention_output = self.attention(self.layer_norm1(hidden_states), method_name)
        layernorm_2_inputs = attention_output + hidden_states

        # FFN.
        mlp_output = self.mlp(self.layer_norm2(layernorm_2_inputs), method_name)
        output = layernorm_2_inputs + mlp_output
        return output


class Transformer(nn.Module):
    """Transformer model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(i, config)
            for i in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x, method_name: str | list):
        """
        前向传播
        
        参数:
            x: 输入张量
            method_name: 可以是字符串（所有层使用相同策略）或策略列表（每层不同策略）
        
        返回:
            模型输出logits
        """
        # 检查method_name类型
        if not isinstance(method_name, str) and not isinstance(method_name, list):
            print(f"nooooooooooooooooooooooooooooooooooo!{method_name}")
            raise TypeError("method_name必须是字符串或字符串列表")
            
        # 如果是列表，检查长度是否匹配层数
        if isinstance(method_name, list):
            if len(method_name) != len(self.layers):
                raise ValueError(f"策略列表长度({len(method_name)})必须匹配层数({len(self.layers)})")
        
        hidden_states = self.token_emb(x)
        
        # 处理不同策略输入方式
        if isinstance(method_name, str):
            # 所有层使用相同策略
            for layer in self.layers:
                hidden_states = layer(hidden_states, method_name)
        else:
            # 每层使用不同策略
            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, method_name[i])
        
        hidden_states = self.ln_f(hidden_states)
        self.self_head = self.head(hidden_states)
        logits = self.self_head
        return logits
