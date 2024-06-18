import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

class Encoder(nn.Module):
    """Encoder通常作为整个网络的入口，接收state数据，将其处理成[batch_size, n]的向量
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self._name = None

    def forward(self,
                inputs: Union[torch.Tensor],
                training: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Parameters
        ----------
        inputs : Union[torch.Tensor]
            特征图，维度[batch_size, ...]

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            outputs :
                维度为[batch_size, n]
            embeddings :
                这是一个可选返回，可以返回一个Tensor, 也可以返回None。比如说当输入是一张
                图像的时候，维度为[batch_size, height, width, channel],
                不仅想返回outputs，还想输出一个保留空间信息的embedding。
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name_: str):
        self._name = name_

    def __str__(self) -> str:
        return self.name
