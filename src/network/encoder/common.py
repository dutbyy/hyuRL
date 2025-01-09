import torch
from torch import nn
from typing import List, Tuple, Union
from hyuRL.src.network.encoder.encoder import Encoder
from hyuRL.src.network.layer.active_tool import make_active_layer


class CommonEncoder(Encoder):
    r"""用于处理环境中的统计特征信息的编码器
    CommonEncoder

    `Args`:
        `in_features` :        输入特征维度
        `hidden_layer_sizes`:  隐藏层大小

    `Shape`:
        - `Input`:  [batch, in_feature]
        - `Output`: [batch, output_size]

    `Example`::
        >>> encoder = CommonEncoder(in_feature=16, hidden_layer_sizes=[128, 128])
        >>> inputs = torch.ones(size=[128, 10])
        >>> output, _ = encoder(inputs)
        >>> print(output.shape)
        torch.Size([128, 128])
    """

    def __init__(self, in_features, hidden_layer_sizes: List[int], output_size=256, activation='relu'):

        super().__init__()
        layers = []
        layer_sizes = [in_features] + hidden_layer_sizes + [output_size]
        # 为后续层添加线性层、ReLU和LayerNorm
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(make_active_layer(activation))
            layers.append(nn.LayerNorm(out_f))
        self._dense_sequence = nn.Sequential(*layers)

    def forward(
        self, inputs: Union[torch.Tensor], training: bool = False
    ) -> Tuple[torch.Tensor, None]:

        return self._dense_sequence(inputs), None
