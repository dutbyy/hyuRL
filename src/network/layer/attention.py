import torch
from torch import nn
from typing import List, Any, Tuple, Dict


# 点积模型
class DotProdcutAttentionScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]):
        q, k = inputs
        score = torch.matmul(q, k.transpose(-2, -1))
        return score


# 缩放点积模型
class ScaledDotProdcutAttentionScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]):
        q, k = inputs
        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        scaled_attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(dk)
        return scaled_attention_scores


# 加法模型
class AdditiveAttentionScore(nn.Module):
    def __init__(self, d_q: int, d_k: int, hidden_size: int = 64):
        super().__init__()
        self.W_q = nn.Linear(d_q, hidden_size, bias=False)
        self.W_k = nn.Linear(d_k, hidden_size, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, inputs) -> torch.Tensor:
        query, key = inputs
        # 线性变换
        transformed_query = self.W_q(query)  # (batch_size, seq_len_q, hidden_size)
        transformed_key = self.W_k(key)  # (batch_size, seq_len_k, hidden_size)

        # 计算加法注意力分数
        transformed_query = transformed_query.unsqueeze(2)
        transformed_key = transformed_key.unsqueeze(1)
        scores = torch.tanh(transformed_query + transformed_key)
        scores = torch.matmul(scores, self.v).squeeze(-1)

        return scores


# 双线性模型
class BilinearAttentionScore(nn.Module):
    def __init__(self, d_q: int, d_k: int):
        super(BilinearAttentionScore, self).__init__()
        self.W = nn.Parameter(torch.randn(d_q, d_k))
        nn.init.xavier_uniform_(self.W)

    def forward(self, inputs) -> torch.Tensor:
        query, key = inputs
        transformed_query = torch.matmul(query, self.W)
        scores = torch.matmul(transformed_query, key.transpose(-1, -2))

        return scores



def attention_score_model(name: str, **args):
    name2score = {
        "dot_product": DotProdcutAttentionScore,
        "scale_dot_product": ScaledDotProdcutAttentionScore,
        "add": AdditiveAttentionScore,
        "binary": BilinearAttentionScore,
    }
    cls = name2score.get(name)
    return cls(**args)


def test_dot():
    query = torch.rand(size=[128, 32, 512])
    key = torch.rand(size=[128, 64, 512])
    score = attention_score_model("dot_product")((query, key))
    print(score.shape)


def test_scaled():
    query = torch.rand(size=[128, 32, 512])
    key = torch.rand(size=[128, 64, 512])
    score = attention_score_model("scale_dot_product")((query, key))
    print(score.shape)


def test_add():
    score_model = attention_score_model("add", d_q=32, d_k=64)
    query = torch.rand(size=[128, 32, 32])
    key = torch.rand(size=[128, 64, 64])
    score = score_model((query, key))
    print(score.shape)


def test_bin():
    score_model = attention_score_model("binary", d_q=32, d_k=64)
    score_model = BilinearAttentionScore(32, 64)
    query = torch.rand(size=[128, 32, 32])
    key = torch.rand(size=[128, 64, 64])
    score = score_model((query, key))
    print(score.shape)


def test_multi():
    att_m = MultiHeadAttention(32, 64, 256, 4, 4)
    query = torch.rand(size=[128, 128, 32])
    key = torch.rand(size=[128, 128, 64])
    value = torch.rand(size=[128, 128, 256])
    att = att_m([query, key, value])
    print(att.shape)


if __name__ == "__main__":
    test_multi()
    test_dot()
    test_scaled()
    test_add()
    test_bin()
