import torch
import torch.nn as nn
from torch.distributions import Normal
from .decoder import Decoder
CommonLayerSize = 256
class GaussianDecoder(Decoder):
    """
    Decoder for handling continuous actions.
    """

    def __init__(self, n, hidden_layer_sizes, activation='relu'):
        super().__init__()
        # 动作数量
        self.n = n
        # 定义隐藏层序列
        layers = []
        layer_sizes = [CommonLayerSize] + hidden_layer_sizes
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if activation == 'relu':
                layers.append(nn.ReLU())
            # 如果需要，可以添加其他激活函数
        layers.append(nn.Linear(hidden_layer_sizes[-1], n))
        self.dense_sequence = nn.Sequential(*layers)
        # 初始化log_std变量
        self.log_std = nn.Parameter(torch.zeros(n), requires_grad=True)
        # 定义动作嵌入层
        self.action_embedding = nn.Linear(n, CommonLayerSize)

    def forward(self, inputs, action_mask=None, behavior_action=None):
        # 通过隐藏层序列处理输入
        mu = self.dense_sequence(inputs[0])
        if action_mask is not None:
            # 在PyTorch中，连续解码器不支持动作掩码
            raise NotImplementedError("action_mask is not supported in ContinuousDecoder")
        # 定义正态分布
        std = torch.exp(self.log_std)
        distribution = Normal(mu, std)

        if behavior_action is None:
            behavior_action = distribution.sample()
        print("behavior_action: ", behavior_action)
        # 获取行为动作的嵌入表示
        behavior_action_embedding = self.action_embedding(behavior_action)
        # 计算自回归嵌入，结合行为动作嵌入和输入
        auto_regressive_embedding = behavior_action_embedding + inputs[0]
        
        return {
            "mu": mu,
            "log_std": self.log_std.expand_as(mu)
        }, behavior_action, auto_regressive_embedding

    def distribution(self, mu):
        # 返回正态分布，使用mu和log_std参数
        std = torch.exp(self.log_std)
        return Normal(mu, std)
