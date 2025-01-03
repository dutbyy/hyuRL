import torch.nn as nn
from torch.distributions import Categorical

from hyuRL.src.network.decoder.decoder import Decoder
from hyuRL.src.network.layer.active_tool import make_active_layer

class CategoricalDecoder(Decoder):
    """
    Decoder for handling discrete actions.
    Args:
        n (_type_): 离散动作空间
        in_features (int): 输入特征长度
        hidden_layer_sizes (_type_): 隐藏层大小
        activation (str, optional): 激活函数. Defaults to "relu".
        temperature (float, optional): _description_. Defaults to 1.0.
    """
    def __init__(self, in_features, n, hidden_layer_sizes, activation="relu", temperature=1.0):
        super().__init__()
        self._n = n
        self._temperature = temperature
        layers = nn.ModuleList()

        layer_sizes = [in_features] + hidden_layer_sizes
        # 为后续层添加线性层、ReLU和LayerNorm
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(make_active_layer(activation))

        layers.append(nn.Linear(layer_sizes[-1], n))
        self._dense_sequence = nn.Sequential(*layers)
        self.embedding_vocabulary = nn.Embedding(self._n, in_features)

    def forward(self, inputs, action_mask=None, behavior_action=None):
        logits = self._dense_sequence(inputs[0])
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        # print(f"categorical decoder logits : {logits}")
        distribution = Categorical(logits=logits / self._temperature)

        # 如果没有提供行为动作，则从分布中采样一个动作
        if behavior_action is None:
            behavior_action = (
                distribution.sample()
            )  # Categorical本身就不可微, 不需要detach
        behavior_action_embedding = self.embedding_vocabulary(behavior_action)

        # 计算自回归嵌入，结合行为动作嵌入和输入
        auto_regressive_embedding = behavior_action_embedding + inputs[0]
        return logits, behavior_action, auto_regressive_embedding

    def distribution(self, logits):
        # print(f"dist logits is {logits}")
        return Categorical(logits=logits / self._temperature)
