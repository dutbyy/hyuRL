import torch
import torch.nn as nn
from typing import List
CommonLayerSize = 256

class ValueApproximator(nn.Module):
    """可以输出 V(s) 的 value network

    Parameters
    ----------
    hidden_layer_sizes : List
        Value network 的隐藏层大小
    """

    def __init__(self, hidden_layer_sizes: List, activation='relu'):
        super(ValueApproximator, self).__init__()
        layers = []
        
        layer_sizes = [CommonLayerSize] + hidden_layer_sizes 
        # 为后续层添加线性层、ReLU和LayerNorm
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if activation == 'relu':
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        self._dense_sequence = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算 V(s) 的值

        Parameters
        ----------
        inputs : torch.Tensor
            Aggregator 输出的中间层 Embedding

        Returns
        -------
        torch.Tensor
            Value 的值的预测
        """
        value = self._dense_sequence(inputs)
        value = torch.squeeze(value, -1)
        return value


if __name__ == '__main__':
    vnet = ValueApproximator([128, 64])
    a = torch.rand(2,128)
    print(vnet(a))