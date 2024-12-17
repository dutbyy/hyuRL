import torch
import torch.nn as nn
from torch.nn import init
from typing import List, Tuple, Union
from .encoder import Encoder
def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class ResnetBlock(nn.Module):
    def __init__(self, channels:int = 64, bn:bool = False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(channels, channels, 3, padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, 3, padding="same"))
        self._net_sequence = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self._net_sequence(inputs)
        return nn.relu(inputs+outputs)


class SpatialEncoder(Encoder):
    """用于处理环境中的统计特征信息的编码器
    """

    def __init__(self, in_features, channel_num, output_size, down_samples=None, res_block_num=4):
        """初始化 SpatialEncoder

        Parameters
        ----------
        in_features: int
        channel_num: int
        output_size: int
            隐藏层输出神经元数量
        """
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_features, channel_num))
        layers.append(nn.ReLU())
        if down_samples:
            for (filters, kernel_size, strides, padding) in down_samples:
                layers.append(nn.Conv2d(channel_num, filters, kernel_size, strides, padding))
                layers.append(nn.ReLU())
                channel_num = filters
        if res_block_num:
            for _ in range(res_block_num):
                layers.append(ResnetBlock(channel_num))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(channel_num, output_size))
        layers.append(nn.ReLU())

        self._dense_sequence = nn.Sequential(*layers)
        self._dense_sequence.apply(init_weights)

    def forward(self,
                inputs: Union[torch.Tensor],
                training: bool = False) -> Tuple[torch.Tensor, None]:
        """
        Parameters
        ----------
        inputs : Union[torch.Tensor]
            common 特征

        Returns
        -------
        Tuple[torch.Tensor, None]
            编码后的特征, 保留统计信息的embedding
        """

        return self._dense_sequence(inputs), None

if __name__ == '__main__':
    a = CommonEncoder(12, [64, 12, 32])
    print(a)

