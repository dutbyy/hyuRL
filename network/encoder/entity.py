import torch
import torch.nn as nn
from typing import Tuple, Union

class EntityEncoder(nn.Module):
    """用于处理环境中实体特征的编码器
    """

    def __init__(self, in_features, hidden_layer_sizes, transformer=None, pooling=None):
        super(EntityEncoder, self).__init__()
        self._pooling = pooling
        self._transformer = transformer
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
        # outputs = self._pooling(entity_embeddings, inputs_len)
        
        return outputs, entity_embeddings

# 注意：这里省略了Mask和Pooling的实现，因为它们需要根据具体情况来定制。

if __name__ == '__main__':
    a = EntityEncoder(24, [12, 24])
    print(a)