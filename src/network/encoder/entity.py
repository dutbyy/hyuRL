import torch
import torch.nn as nn
from typing import Tuple, Union
from .encoder import Encoder

def init_weights(m):
    from torch.nn import init
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class MaxPooling(nn.Module):
    def __init__(self, length, dim=1,):
        super().__init__()
        self.pooling = torch.nn.MaxPool1d(length)

    def forward(self, x):
        y = self.pooling(x.permute(0, 2, 1))
        
        return torch.flatten(y, 1)
    
class EntityEncoder(Encoder):
    """用于处理环境中实体特征的编码器
    `Args`:
        `in_features` :         输入特征维度
        `hidden_layer_sizes`:   隐藏层大小
        `transformer`:          transformer
        `pooling`:              pooling
        
    `Shape`:
        - `Input`:  [batch, in_feature]
        - `Output`: [batch, output_size]

    `Example`::
        >>> encoder = EntityEncoder(in_feature=16, hidden_layer_sizes=[128, 128])
        >>> inputs = torch.ones(size=[128, 10])
        >>> output, _ = encoder(inputs)
        >>> print(output.shape)
        torch.Size([128, 128])
    """

    def __init__(self, length, in_features, hidden_layer_sizes, transformer=None, pooling=None):
        super(EntityEncoder, self).__init__()
        self._transformer = transformer
        self._pooling = pooling if pooling else MaxPooling(length)
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
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : Union[torch.Tensor]
            实体特征

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            outputs: 编码后的实体特征
            entity_embeddings: 保留实体信息的embedding
        """
        inputs_len = inputs.size(1)  # 假设inputs的形状为 [batch_size, seq_len, feature_size]
        
        entity_embeddings = self._dense_sequence(inputs)
        
        # 应用Mask，这里需要自定义一个Mask函数或者使用PyTorch的PackedSequence来处理变长序列
        # entity_embeddings = Mask(entity_embeddings, inputs_len)
        
        if self._transformer is not None:
            entity_embeddings = self._transformer(entity_embeddings)
        
        # 应用Pooling，这里需要自定义一个Pooling类或者使用PyTorch的池化层来实现相应功能
        # outputs = torch.ones([100,1])
        if self._pooling:
            outputs = self._pooling(entity_embeddings)
        
        return outputs, entity_embeddings


