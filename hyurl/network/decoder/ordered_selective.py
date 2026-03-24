from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyurl.network.layer.distribution import OrderedMultiSelective
from .decoder import Decoder, decoder_pooling
from hyurl.network.layer.ptrnet import PointerNetwork, indices_to_binary


class OrderedMultipleSelectiveDecoder(Decoder):
    """用于处理有序选择多个单位的解码器
    """

    def __init__(self, max_count: int, pooling=decoder_pooling.MeanMax()):
        super(OrderedMultipleSelectiveDecoder, self).__init__()
        self.ptr_net = PointerNetwork(max_count)
        self.pooling = pooling
        self.dense_1 = None
        self.dense_2 = None

    def distribution(self, logits):
        return OrderedMultiSelective(logits)

    def build(self, input_shape):
        output_size = input_shape[0][-1]
        self.dense_1 = nn.Linear(output_size, output_size * 2)
        self.dense_2 = nn.Linear(output_size * 2, output_size)
        super(OrderedMultipleSelectiveDecoder, self).build(input_shape)

    def forward(self,
                inputs: List[torch.Tensor],
                action_mask: np.ndarray = None,
                behavior_action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs, source_embeddings = inputs
        logits, action = self.ptr_net({
            'auto_embed': inputs,
            'source': source_embeddings
        },
                                      units_selected_ph=behavior_action,
                                      select_unit_mask=action_mask)

        if behavior_action is None:
            behavior_action = action
        behavior_action = behavior_action.to(torch.int32)
        behavior_action = indices_to_binary(behavior_action, max_len=source_embeddings.size(1))

        selected_entities_embedding = self.pooling(source_embeddings, behavior_action)

        selected_entities_embedding = F.relu(self.dense_1(selected_entities_embedding))
        selected_entities_embedding = self.dense_2(selected_entities_embedding)
        auto_regressive_embedding = inputs + selected_entities_embedding

        return logits, action, auto_regressive_embedding