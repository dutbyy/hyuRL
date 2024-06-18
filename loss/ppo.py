import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOLoss(nn.Module):
    def __init__(self, clip_epsilon=0.2, value_clip=0.5, value_coef=0.5, entropy_coef=0.01):
        """
        初始化 PPOLoss 模块。

        Args:
            clip_epsilon (float): 用于裁剪 surrogate loss 的参数。
            value_clip (float): 用于裁剪值函数预测值的参数。
            value_coef (float): 值函数损失的系数。
            entropy_coef (float): 熵损失的系数。
        """
        super(PPOLoss, self).__init__()
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def forward(self, advantages, old_probs, new_probs, values):
        """
        计算 PPO 损失。

        Args:
            advantages (torch.Tensor): 优势估计。
            old_probs (torch.Tensor): 旧的动作概率。
            new_probs (torch.Tensor): 新的动作概率。
            values (torch.Tensor): 预测的状态值。

        Returns:
            torch.Tensor: 总的 PPO 损失。
        """
        # 计算 surrogate loss
        ratio = new_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 裁剪值函数预测值
        clipped_values = torch.clamp(values, -self.value_clip, self.value_clip)

        # 计算值函数损失
        value_loss = F.mse_loss(clipped_values, advantages)

        # 计算熵损失
        entropy_loss = -(new_probs * torch.log(new_probs)).mean()

        # 总的损失
        loss = surrogate_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        return loss

def test():
    # 示例用法：
    advantages = torch.tensor([0.1, 0.2, 0.3])
    old_probs = torch.tensor([0.4, 0.5, 0.6])
    new_probs = torch.tensor([0.45, 0.55, 0.65])
    values = torch.tensor([0.7, 0.8, 0.9])

    loss_fn = PPOLoss()
    loss = loss_fn(advantages, old_probs, new_probs, values)
    print("总损失:", loss.item())

