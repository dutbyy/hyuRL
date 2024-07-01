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
        # TRPO 提出的 代理 loss
        # error: 直观理解: 原有的梯度是 ▽logp * Adv, 为了便于加入约束, 修改为▽(p-oldp)*Adv [ p和logp的作为优化目标应该是等价的, 而oldp是常数和当前优化的策略pi无关]
        # 优化目标为 J(θ) = E(π/π_old * adv_old)
        # 为了限制参数更新的距离(clip), 通过loss加入clip 约束梯度更新
        # ratio:  exp(logp-logpold) = p/oldp  表示新旧策略的差距 clip > 1 表示当前策略比之前更大概率选择当前动作
        # adv表示当前动作的好坏(相比期望)
        # adv为正: 
        #     ratio > 1 + clip, clip掉 (动作好的时候, 避免太容易选)
        #     ratio < 1 - clip, 不进行clip
        # adv为负: 
        #     ratio < 1 - clip, clip掉 (动作不好的时候, 避免完全不选)
        #     ratio > 1 + clip, 不进行clip 
        # 计算 surrogate loss
        ratio =  torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon)
        surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = surrogate_loss.mean()
        
        clipped_mask = (- ratio * advantage != surrogate_loss).float()
        
        # 裁剪值函数预测值       
        # 此处的clip是希望避免value的更新太激进
        # v_old, -> v_target ; 如果v_pred 在两者之间, 且v_pred的距和v_old的距离已经超过clip，则需要将其进行clip 这个时候 clip loss > pred loss        
        value_pred_clip = old_value + torch.clamp(value - old_value, -self._value_clip, self._value_clip)
        value_loss1 = (value - target_value).pow(2)
        value_loss2 = (value_pred_clip - target_value).pow(2)
        clipped_loss = torch.max(value_loss1, value_loss2)
        value_loss = 0.5 * clipped_loss.mean() 

        value_loss = self._value_coef * value_loss
        entropy_loss = - self._entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss
        return loss, policy_loss, value_loss, entropy_loss, ratio, clipped_mask
    
    
def test():
    # 示例用法：
    advantages = torch.tensor([0.1, 0.2, 0.3])
    old_probs = torch.tensor([0.4, 0.5, 0.6])
    new_probs = torch.tensor([0.45, 0.55, 0.65])
    values = torch.tensor([0.7, 0.8, 0.9])

    loss_fn = PPOLoss()
    loss = loss_fn(advantages, old_probs, new_probs, values)
    print("总损失:", loss.item())

