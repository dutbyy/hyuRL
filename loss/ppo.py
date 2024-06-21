import torch
import torch.nn as nn
import torch.nn.functional as F


def lfPPO(old_log_prob, log_prob, advantage, old_value, value, target_value, entropy):
    pass 

class PPOLoss(nn.Module):
    def __init__(self, clip_epsilon=0.2, value_clip=5, value_coef=1, entropy_coef=0.01):
        """
        初始化 PPOLoss 模块。

        Args:
            clip_epsilon (float): 用于裁剪 surrogate loss 的参数。
            value_clip (float): 用于裁剪值函数预测值的参数。
            value_coef (float): 值函数损失的系数。
            entropy_coef (float): 熵损失的系数。
        """
        super().__init__()
        self._clip_epsilon = clip_epsilon
        self._value_clip = value_clip
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef

    def forward(self, old_log_prob, log_prob, advantage, old_value, value, target_value, entropy):
        """
        计算 PPO 损失。

        Args:
            advantage (torch.Tensor): 优势估计。
            old_probs (torch.Tensor): 旧的动作概率。
            new_probs (torch.Tensor): 新的动作概率。
            values (torch.Tensor): 预测的状态值。

        Returns:
            torch.Tensor: 总的 PPO 损失。
        """
        # 计算 surrogate loss
        ratio =  torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon)
        surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        policy_loss = torch.mean(surrogate_loss)
 
        # 裁剪值函数预测值       
        value_pred_clip = old_value + torch.clamp(value - old_value, -self._value_clip, self._value_clip)
        value_loss1 = (value - target_value).pow(2)
        value_loss2 = (value_pred_clip - target_value).pow(2)
        value_loss = 0.5 * torch.mean(torch.min(value_loss1, value_loss2))
        # value_diff1 = torch.sum(torch.max(value_loss1, value_loss2) - value_loss1)
        # value_diff2 = torch.sum(torch.max(value_loss1, value_loss2) - value_loss2)
        # if (torch.sum(value_loss1-value_loss2)) > 1:
        #     print(f"value_diff1 is {value_diff1}")
        #     print(f"value_diff2 is {value_diff2}")
        value_loss = self._value_coef * value_loss
        entropy_loss = - self._entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss
        # print(f"policy loss : {policy_loss}, value_loss : {value_loss}, entropy: {entropy}, loss: {loss}")
        return loss, policy_loss, value_loss, entropy_loss, ratio
    
    
def test():
    # 示例用法：
    advantages = torch.tensor([0.1, 0.2, 0.3])
    old_probs = torch.tensor([0.4, 0.5, 0.6])
    new_probs = torch.tensor([0.45, 0.55, 0.65])
    values = torch.tensor([0.7, 0.8, 0.9])

    loss_fn = PPOLoss()
    loss = loss_fn(advantages, old_probs, new_probs, values)
    print("总损失:", loss.item())

