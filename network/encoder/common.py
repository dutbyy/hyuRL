import torch
import torch.nn as nn
from torch.nn import init
from typing import List, Tuple, Union

def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class CommonEncoder(nn.Module):
    """用于处理环境中的统计特征信息的编码器
    """

    def __init__(self, in_features, hidden_layer_sizes: List[int]):
        """初始化 CommonEncoder

        Parameters
        ----------
        hidden_layer_sizes : List[int]
            隐藏层输出神经元数量
        """
        super().__init__()
        layers = []
        layer_sizes = [in_features] + hidden_layer_sizes
        # 为后续层添加线性层、ReLU和LayerNorm
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_f))
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
            编码后的特征；保留统计信息的embedding
        """
        # print('common encoder', next(self.parameters()).is_cuda)
        return self._dense_sequence(inputs), None

if __name__ == '__main__':
    a = CommonEncoder(12, [64, 12, 32])
    print(a)