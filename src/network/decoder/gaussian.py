import torch
from torch import nn
from torch.distributions import Normal

from hyuRL.src.network.decoder.decoder import Decoder

class GaussianDecoder(Decoder):
    """连续动作空间

    Args:
        n (_type_): 动作个数
        hidden_layer_sizes (_type_): 隐藏层大小列表
        activation (str, optional): 激活函数. Defaults to 'relu'.
    """
    def __init__(self, n, in_features, hidden_layer_sizes, activation='relu'):
        super().__init__()
        # 动作数量
        self.n = n
        # 定义隐藏层序列
        layers = []
        layer_sizes = [in_features] + hidden_layer_sizes
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if activation == 'relu':
                layers.append(nn.ReLU())
            # 如果需要，可以添加其他激活函数
            elif activation == 'tanh':
                layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layer_sizes[-1], n))
        self.dense_sequence = nn.Sequential(*layers)
        # 初始化log_std变量
        self.log_std = nn.Parameter(torch.zeros(n, dtype=torch.float32), requires_grad=True)
        # 定义动作嵌入层
        self.action_embedding = nn.Linear(n, in_features)

    def forward(self, inputs, action_mask=None, behavior_action=None):
        # 通过隐藏层序列处理输入
        inputs, embedding = inputs
        mu = self.dense_sequence(inputs)
        if action_mask is not None:
            # 在PyTorch中，连续解码器不支持动作掩码
            raise NotImplementedError("action_mask is not supported in ContinuousDecoder")
        # 定义正态分布
        std = torch.exp(self.log_std)
        distribution = Normal(mu, std)

        if behavior_action is None:
            # behavior_action = distribution.sample()
            behavior_action = distribution.sample().detach()
        # 获取行为动作的嵌入表示
        behavior_action_embedding = self.action_embedding(behavior_action)
        # 计算自回归嵌入，结合行为动作嵌入和输入
        auto_regressive_embedding = behavior_action_embedding + inputs
        return {
            "mu": mu,
            "log_std": self.log_std.expand_as(mu)
        },  behavior_action, auto_regressive_embedding

    def distribution(self, mu):
        # 返回正态分布，使用mu和log_std参数
        std = torch.exp(self.log_std)
        return Normal(mu, std)
