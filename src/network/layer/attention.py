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
        scores = torch.tanh(
            transformed_query.unsqueeze(2) + transformed_key.unsqueeze(1)
        )
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
        batch_size, seq_len_q, d_q = query.size()
        _, seq_len_k, d_k = key.size()
        transformed_query = torch.matmul(query, self.W)
        scores = torch.matmul(transformed_query, key.transpose(-1, -2))

        return scores


class MultiHeadAttention(nn.Module):

    def __init__(self, d_q, d_k, d_v, num_heads: int, head_size: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        d_model = num_heads * head_size
        self.wq = nn.Linear(d_q, d_model, bias=False)
        self.wk = nn.Linear(d_k, d_model, bias=False)
        self.wv = nn.Linear(d_v, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        
    def split_heads(self, x:torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_size).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, head_size)
        """
        x = x.reshape([batch_size, -1, self.num_heads, self.head_size])
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        q, k, v = inputs
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, head_size)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, head_size)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, head_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_scores = ScaledDotProdcutAttentionScore()((q, k))
        # if v_len is not None:
        #     scaled_attention_scores = Mask(
        #         scaled_attention_scores, v_len, mode="add", seq_axis=-1
        #     )

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = torch.softmax(scaled_attention_scores, axis=-1)  # (..., seq_len_q, seq_len_k)

        # (..., seq_len_q, depth_v)
        scaled_attention = torch.matmul(attention_weights, v)

        # (batch_size, seq_len_q, num_heads, head_size)
        scaled_attention.permute([0, 2, 1, 3])

        # concat multiple heads
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape((-1, scaled_attention.shape[1], self.num_heads * self.head_size))

        # (batch_size, seq_len_q, d_model)
        outputs = self.linear(concat_attention)
        # if q_len is not None:
        #     outputs = Mask(outputs, q_len, mode="mul", seq_axis=1)
        return outputs


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
    score = attention_score_model('dot_product')([query, key])
    print(score.shape)


def test_scaled():
    query = torch.rand(size=[128, 32, 512])
    key = torch.rand(size=[128, 64, 512])
    score = attention_score_model('scale_dot_product')([query, key])
    print(score.shape)


def test_add():
    score_model = attention_score_model('add', d_q=32, d_k=64)
    query = torch.rand(size=[128, 32, 32])
    key = torch.rand(size=[128, 64, 64])
    score = score_model([query, key])
    print(score.shape)


def test_bin():
    score_model = attention_score_model('binary', d_q=32, d_k=64)
    score_model = BilinearAttentionScore(32, 64)
    query = torch.rand(size=[128, 32, 32])
    key = torch.rand(size=[128, 64, 64])
    score = score_model([query, key])
    print(score.shape)

def test_multi():
    att_m = MultiHeadAttention(32, 64, 256, 4, 4)
    query = torch.rand(size=[128, 128, 32])
    key =   torch.rand(size=[128, 128, 64])
    value = torch.rand(size=[128, 128, 256])
    att = att_m([query, key, value])
    print(att.shape)
    
    
if __name__ == "__main__":
    test_multi()
    test_dot()
    test_scaled()
    test_add()
    test_bin()
