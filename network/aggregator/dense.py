from typing import List, Tuple
import torch
import torch.nn as nn

class DenseAggregator(nn.Module):

    def __init__(self, hidden_layer_sizes: List[int], output_size: int, in_features: int=128):
        super(DenseAggregator, self).__init__()
        layers = []
        
        layer_sizes = [in_features] + hidden_layer_sizes 
        # 为后续层添加线性层、ReLU和LayerNorm
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        self._dense_sequence = nn.Sequential(*layers)

    def forward(self,
                inputs: List[torch.Tensor],
                initial_state=None,
                seq_len: int = 1, training:bool=False) -> Tuple[torch.Tensor, None]:
        del initial_state, seq_len  # Unused by forward
        concat_features = torch.cat(inputs, dim=-1)
        outputs = self._dense_sequence(concat_features)
        return outputs, None
