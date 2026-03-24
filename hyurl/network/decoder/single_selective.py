from __future__ import annotations

import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from typing import List, Tuple

from .decoder import Decoder
from hyurl.network.layer.attention import attention_score_model
from hyurl.network.layer.active_tool import make_active_layer


class MeanMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, entities_embeddings: torch.Tensor, selected_entity_one_hot: torch.Tensor
    ) -> torch.Tensor:

        selected_entity_one_hot = selected_entity_one_hot.unsqueeze(
            -1
        )  # (batch_size, max_num, 1)

        # 选择被选中的实体的嵌入
        selected_embeddings = entities_embeddings * selected_entity_one_hot # (batch_size, max_num, depth)

        # 计算每个样本中被选中的实体数量，避免除以 0
        selected_count = torch.clamp(torch.sum(selected_entity_one_hot, dim=1), min=1.0 )  # (batch_size, 1)

        # 计算 mean pooling
        selected_mean_embeddings = (torch.sum(selected_embeddings, dim=1) / selected_count)  # (batch_size, depth)

        # 计算 max pooling
        selected_max_embeddings, _ = torch.max(selected_embeddings, dim=1)  # (batch_size, depth)

        # 拼接 mean 和 max 的结果
        selected_mean_max_embeddings = torch.cat([selected_mean_embeddings, selected_max_embeddings], dim=-1)  # (batch_size, depth * 2)

        return selected_mean_max_embeddings

def get_mask(inputs: torch.Tensor) -> torch.Tensor:
    # [batch_size, ..., dim] -> [batch_size, ...]
    # max_abs_inputs = torch.max(torch.abs(inputs), dim=-1).values  # (batch_size, seq_len)
    # mask = torch.gt(max_abs_inputs, 0.0)  # (batch_size, seq_len)
    # mask = mask.to(dtype=torch.float32)  # (batch_size, seq_len)

    mask = (torch.max(torch.abs(inputs), dim=-1).values > 0.0).float()

    return mask


def apply_mask(inputs, mask=None, mode="mul"):
    if mask is None:
        return inputs
    if mode == "mul":
        return inputs * mask
    elif mode == "add":
        return inputs - (1 - mask.float()) * 1e12
    else:
        raise Exception(f"Unsupport Maks Mode {mode}")

class SingleSelectiveDecoder(Decoder):
    """用于处理单个单位选择的解码器

    Args:
        in_features (_type_): 输入特征维度
        attention_size (int, optional): 注意力隐藏层大小. Defaults to 64.
    """
    def __init__(self, in_features:int, attention_size: int = 64, activation='relu'):

        super(SingleSelectiveDecoder, self).__init__()
        self._add_attention = attention_score_model(
            "add", d_q=in_features, d_k=in_features, hidden_size=attention_size
        )
        self.linear_seq = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            make_active_layer(activation),
            nn.Linear(in_features, in_features),
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        action_mask: np.ndarray = None,
        behavior_action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs, source_embeddings = inputs  # (batch_size, in_features), (batch_size, seq_len, in_features)
        seq_len = source_embeddings.shape[1]
        query = inputs.unsqueeze(1)         # (batch_size, 1,       in_features)
        key = source_embeddings             # (batch_size, seq_len, in_features)
        attention_score = self._add_attention((query, key))     # (batch_size, 1, seq_len)

        logits = attention_score.squeeze(1)                     # (batch_size, seq_len)

        # print(source_embeddings)
        mask = get_mask(source_embeddings)                      # (batch_size, seq_len) 0/1
        logits = apply_mask(logits, mask, mode="add")           # (batch_size, seq_len)
        if action_mask is not None:
            logits = apply_mask(logits, action_mask, mode="add")

        distribution = self.distribution(logits=logits)
        if behavior_action is None:
            behavior_action:torch.Tensor = distribution.sample()
        behavior_action = behavior_action.long()                                    # (batch_size,)
        behavior_action_one_hot = nn.functional.one_hot(behavior_action, seq_len)   # (batch_size, seq_len)
        pooling = MeanMax()
        selected_embedding = pooling(source_embeddings, behavior_action_one_hot)    # (batch_size, in_features * 2)
        selected_embedding = self.linear_seq(selected_embedding)                    # (batch_size, in_features)
        auto_regressive_embedding = selected_embedding + inputs
        return logits, behavior_action, auto_regressive_embedding

    def distribution(self, logits):
        return Categorical(logits=logits)


if __name__ == "__main__":
    decoder = SingleSelectiveDecoder(256, 64)
    inputs = torch.rand(32, 256)
    source_embeddings = torch.rand(32, 16, 256)

    logits, behavior_action, auto_regressive_embedding = decoder(
        (inputs, source_embeddings)
    )
    print(logits.shape)
    print(behavior_action.shape)
    print(auto_regressive_embedding.shape)
