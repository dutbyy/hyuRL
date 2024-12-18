from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .decoder import Decoder


class SingleSelectiveDecoder(Decoder):
    """用于处理单个单位选择的解码器
    """

    def __init__(self,in_features, attention_size: int = 64, temperature: float = 1.0):
        super(SingleSelectiveDecoder, self).__init__()
        self._temperature = temperature
        self._wq = nn.Linear(in_features, attention_size)
        self._wk = nn.Linear(in_features, attention_size)
        self._concat_attention_score = AttentionScoreFactory.get("concat")
        self._dense_1, self._dense_2 = None, None

    def distribution(self, logits):
        return Categorical(logits=logits, temperature=self._temperature)


    def call(self,
             inputs: List[torch.Tensor],
             action_mask: np.ndarray = None,
             behavior_action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        inputs, source_embeddings = inputs
        query = self._wq(inputs).unsqueeze(1)  # PyTorch 中使用 unsqueeze 而不是 expand_dims
        key = self._wk(source_embeddings)
        logits = self._concat_attention_score((query, key))
        logits = logits.squeeze(dim=1)  # PyTorch 中使用 squeeze 并指定 dim

        mask = get_mask(source_embeddings)
        logits = apply_mask(logits, mask, mode='add')

        if action_mask is not None:
            logits = logits - (1 - action_mask.float()) * 1e12  # PyTorch 中使用 float() 而不是 cast

        distribution = self.distribution(logits)

        if behavior_action is None:
            behavior_action = distribution.sample()
        behavior_action = behavior_action.long()  # PyTorch 中使用 long() 而不是 int32
        behavior_action_one_hot = F.one_hot(behavior_action, num_classes=source_embeddings.shape[1]).squeeze(1)

        pooling = nn.MeanMax()
        selected_embedding = pooling(source_embeddings, behavior_action_one_hot)
        selected_embedding = F.relu(self._dense_1(selected_embedding))  # 使用 F.relu 代替 tf.nn.relu
        selected_embedding = self._dense_2(selected_embedding)
        auto_regressive_embedding = selected_embedding + inputs
        
        
        inputs, source_embeddings = inputs
        query = tf.expand_dims(self._wq(inputs), 1)
        key = self._wk(source_embeddings)
        logits = self._concat_attention_score((query, key))
        logits = tf.squeeze(logits, axis=1)
        mask = get_mask(source_embeddings)
        logits = apply_mask(logits, mask, mode='add')
        if action_mask is not None:
            logits_type = logits.dtype
            logits = logits - tf.multiply(
                tf.constant(1, dtype=logits_type) - tf.cast(action_mask, dtype=logits_type),
                tf.constant(1e12, dtype=logits_type))
        distribution = Categorical(logits, self._temperature)

        if behavior_action is None:
            behavior_action = distribution.sample()
        behavior_action = tf.cast(behavior_action, tf.int32)
        behavior_action_one_hot = tf.one_hot(behavior_action, depth=tf.shape(source_embeddings)[1])
        pooling = decoder_pooling.MeanMax()
        selected_embedding = pooling(source_embeddings, behavior_action_one_hot)
        selected_embedding = self._dense_1(selected_embedding)
        selected_embedding = self._dense_2(selected_embedding)
        auto_regressive_embedding = selected_embedding + inputs

        return logits, behavior_action, auto_regressive_embedding