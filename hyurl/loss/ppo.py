import torch
import torch.nn as nn


class PPOLoss(nn.Module):
    """创建 PPOLoss

    Args:
        clip_epsilon (float): 用于裁剪 surrogate loss 的参数. Defaults to 0.2.
        value_clip (float): 用于裁剪值函数预测值的参数. Defaults to 5.
        value_coef (float): 值函数损失的系数. Defaults to 1.
        entropy_coef (float): 熵损失的系数. Defaults to 0.01.
    """

    def __init__(
        self, clip_epsilon: float = 0.2, value_clip: float = 5.0, value_coef: float = 0.5, entropy_coef: float = 0.01
    ):
        super().__init__()
        self._clip_epsilon = clip_epsilon
        self._value_clip = value_clip
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef

    def forward(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantage: torch.Tensor,
        old_value: torch.Tensor,
        value: torch.Tensor,
        target_value: torch.Tensor,
        entropy: torch.Tensor,
    ):
        """ 计算 PPO 损失。

        Args:
            old_log_prob (torch.Tensor): action在原有策略的输出分布的logp
            log_prob (torch.Tensor): action在当前策略的输出分布的logp
            advantage (torch.Tensor): _description_
            old_value (torch.Tensor): _description_
            value (torch.Tensor): _description_
            target_value (torch.Tensor): _description_
            entropy (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """        """

        Args:
            advantage (torch.Tensor): 优势估计。
            old_log_probs (torch.Tensor): 旧的动作概率。
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
        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_unclip = advantage * ratio
        clipped_ratio = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon)
        policy_loss_clip = advantage * clipped_ratio
        surrogate_loss = torch.min(policy_loss_unclip, policy_loss_clip)

        policy_loss = -surrogate_loss.mean()
        clipped_mask = (policy_loss_unclip != surrogate_loss).float()
        clipped_fraction = clipped_mask.mean()

        # 裁剪值函数预测值, 此处的clip是希望避免value的更新太激进
        # v_old, -> v_target ; 如果v_pred 在两者之间, 且v_pred的距和v_old的距离已经超过clip，则需要将其进行clip 这个时候 clip loss > pred loss
        value_pred_clip = torch.clamp(value, old_value - self._value_clip, old_value + self._value_clip)

        value_loss1 = (value - target_value).pow(2)
        value_loss2 = (value_pred_clip - target_value).pow(2)
        value_loss = self._value_coef * torch.max(value_loss1, value_loss2).mean()

        entropy_loss = -self._entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            ratio_diff = torch.mean((ratio-1.0).abs())

        return loss, policy_loss, value_loss, entropy_loss, ratio_diff, clipped_fraction
