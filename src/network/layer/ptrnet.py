from __future__ import annotations

import torch
from torch import nn

class PointerNetwork(nn.Module):
    def __init__(self, max_cnt):
        super().__init__()
        self.max_cnt = max_cnt

    def forward(x):
        return x

indices_to_binary = lambda x : x