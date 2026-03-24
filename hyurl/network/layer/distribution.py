
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderedMultiSelective(nn.Module):
    """离散随机变量的概率分布（概率质量函数），用于有序多选
    """

    def __init__(self, logits, temperature=1.0):
        super().__init__()
        self._logits = logits
        self._temperature = temperature

    @property
    def logits(self):
        return self._logits

    def sample(self):
        return self._sample_softmax_with_logits(self._logits, self._temperature)

    def negative_logp(self, action):
        action = action.to(dtype=torch.int64)  # 转换为整数类型
        neg_logp = F.cross_entropy(
            input=self._logits.view(-1, self._logits.size(-1)),  # 展平 logits
            target=action.view(-1),  # 展平 action
            reduction='none'
        )
        neg_logp = neg_logp.view_as(action)  # 恢复原始形状
        return torch.sum(neg_logp, dim=-1)  # 沿最后一维求和

    def entropy(self):
        entropy = self._softmax_entropy_with_logits(self._logits)
        return torch.sum(entropy, dim=-1)  # 沿最后一维求和

    def kl(self, other: OrderedMultiSelective):
        kl = self._softmax_kl_with_logits(self._logits, other.logits)
        return torch.sum(kl, dim=-1)  # 沿最后一维求和

    def _sample_softmax_with_logits(self, logits, temperature):
        """从 logits 中采样，使用温度参数控制分布的平滑度"""
        probs = F.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action

    def _softmax_entropy_with_logits(self, logits):
        """计算 softmax 分布的熵"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def _softmax_kl_with_logits(self, logits_p, logits_q):
        """计算两个 softmax 分布之间的 KL 散度"""
        probs_p = F.softmax(logits_p, dim=-1)
        log_probs_p = F.log_softmax(logits_p, dim=-1)
        log_probs_q = F.log_softmax(logits_q, dim=-1)
        kl = torch.sum(probs_p * (log_probs_p - log_probs_q), dim=-1)
        return kl


class UnorderedMultiSelective(nn.Module):
    """离散随机变量的概率分布（概率质量函数），用于无序多选
    """

    def __init__(self, logits, temperature=1.0):
        super().__init__()
        self._logits = logits
        self._temperature = temperature

    @property
    def logits(self):
        return self._logits

    def sample(self):
        probs = torch.sigmoid(self._logits)
        u = torch.rand_like(probs)  # 生成均匀分布的随机数
        action = (probs >= u).float()  # 将布尔值转换为浮点数
        action = action.detach()  # 等价于 tf.stop_gradient
        return action

    def negative_logp(self, action):
        action = action.to(dtype=self._logits.dtype)
        neg_logp = F.binary_cross_entropy_with_logits(
            input=self._logits, target=action, reduction='none'
        )
        neg_logp = torch.sum(neg_logp, dim=-1)  # 沿最后一维求和
        return neg_logp

    def log_prob(self, action):
        action = action.to(dtype=self._logits.dtype)
        neg_logp = F.binary_cross_entropy_with_logits(
            input=self._logits, target=action, reduction='none'
        )
        log_prob = -torch.sum(neg_logp, dim=-1)  # 沿最后一维求和
        return log_prob

    def entropy(self):
        probs = torch.sigmoid(self._logits)
        entropy = F.binary_cross_entropy_with_logits(
            input=self._logits, target=probs, reduction='none'
        )
        entropy = torch.sum(entropy, dim=-1)  # 沿最后一维求和
        return entropy

    def kl(self, other: UnorderedMultiSelective):
        probs = torch.sigmoid(self._logits)
        cross_entropy = F.binary_cross_entropy_with_logits(
            input=other.logits, target=probs, reduction='none'
        )
        entropy = F.binary_cross_entropy_with_logits(
            input=self._logits, target=probs, reduction='none'
        )
        kl = cross_entropy - entropy
        kl = torch.sum(kl, dim=-1)  # 沿最后一维求和
        return kl