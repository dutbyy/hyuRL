import torch
import math
from torch import nn
from typing import List, Tuple, Union

from .encoder import Encoder


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int = 64, bn: bool = False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(channels, channels, 3, padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, 3, padding="same"))
        self._net_sequence = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self._net_sequence(inputs)
        return torch.relu(inputs + outputs)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims)


class SpatialEncoder(Encoder):
    r"""
    创建一个空间特征编码器

    `Args`:
        `in_shape` (List[int]):     空间大小[heigt, width]
        `in_features` (int):        特征维度
        `channel_num` (int):        卷积输出的channel长度
        `output_size` (int):        SpatialEncoder 输出的特征长度
        `down_samples` (List[int], optional):   None or [filters, kernel_size, strides, padding. Defaults to None.
        `res_block_num` (int, optional):  残差块的数量. Defaults to 4.

    `Shape`:
        - `Input`:  [batch, height, width, in_feature]
        - `Output`: [batch, output_size]

    `Example`::
        >>> encoder = SpatialEncoder(in_shape=[16, 16], in_features=10, channel_num=4, output_size=256, down_samples=None, res_block_num=4)
        >>> inputs = torch.ones(size=[128, 16, 16, 10])
        >>> output, _ = encoder(inputs)
        >>> print(output.shape)
        torch.Size([128, 256])
    """

    def __init__(
        self,
        in_shape: List[int],
        in_features: int,
        channel_num: int,
        output_size: int,
        down_samples: List[int] = None,
        res_block_num: int = 4,
    ):
        super().__init__()
        layers = []

        # layers.append(nn.Linear(in_features, channel_num))
        # layers.append(nn.ReLU())
        layers.append(Permute(0, 3, 1, 2))
        pre_shape = in_shape
        print(f"pre_shape is {pre_shape}")
        if down_samples:
            for filters, kernel_size, strides, padding in down_samples:
                if padding == 'same':
                    out_shape = [math.ceil(it/strides) for it in pre_shape]
                    padded_shape = [int(it*strides -1 + kernel_size) for it in out_shape]
                    padding = [ (p1-p2)//2 for p1, p2 in zip(padded_shape, pre_shape)]
                    pre_shape = out_shape
                elif padding == 'valid' or padding == 0:
                    pre_shape = [(pre_shape[0]- kernel_size)//strides + 1 , (pre_shape[1]-kernel_size)//strides + 1 ]
                    print(f"strides is {strides} pre_shape is {pre_shape}")
                layers.append(
                    nn.Conv2d(channel_num, filters, kernel_size, strides, padding)
                )
                layers.append(nn.ReLU())
                channel_num = filters
        if res_block_num:
            for _ in range(res_block_num):
                layers.append(ResnetBlock(channel_num))

        layers.append(Permute(0, 2, 3, 1))
        layers.append(nn.Flatten())
        shape_size = math.prod(pre_shape)
        layers.append(nn.Linear(shape_size * channel_num, output_size))
        layers.append(nn.ReLU())

        self._net_sequence = nn.Sequential(*layers)
        self._net_sequence.apply(init_weights)

    def forward(
        self, inputs: Union[torch.Tensor], training: bool = False
    ) -> Tuple[torch.Tensor, None]:
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

        return self._net_sequence(inputs), None
