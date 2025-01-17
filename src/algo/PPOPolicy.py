from __future__ import annotations
from typing import Dict, Any, Union
from torch import nn
import torch

from hyuRL.src.network.commander import ComplexNetwork
from hyuRL.src.api.net.net import CommanderNetworkConfig
from hyuRL.src.loss.ppo import PPOLoss
from hyuRL.src.tools.common import construct, Summary

def check_gradient_clipping(model, max_grad_norm):
    # 计算梯度范数
    total_norm_before = torch.norm(
        torch.stack(
            [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]
        ),
        2.0,
    )

    # 执行梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 计算裁剪后的梯度范数
    total_norm_after = torch.norm(
        torch.stack(
            [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]
        ),
        2.0,
    )

    # 返回裁剪信息
    return total_norm_before, total_norm_after


class PPOPolicy:
    """PPO策略
    Args:
        network_config (Union[CommanderNetworkConfig, nn.Module]): 决策网络的结构配置
        device (str, optional): 网络使用设备 cuda or cpu. Defaults to "cpu".
        trainning (bool, optional): 是否训练. Defaults to False.
        learning_rate (float, optional): adam的学习率. Defaults to 2.5e-4.
        eps (float, optional): adam的eps. Defaults to 1e-8.
        clip_epsilon (float, optional): ppo clip参数. Defaults to 0.2.
        value_clip (float, optional): value clip, 避免单次reward-td太大导致梯度爆炸. Defaults to 5.0.
        value_coef (float, optional): value loss 权重. Defaults to 0.5.
        entropy_coef (float, optional): entropy loss 权重. Defaults to 0.01.
        max_grad_norm (float, optional): 梯度裁剪最大值. Defaults to 0.5.
        adv_norm (bool, optional): 是否其实用 advantage normalize. Defaults to False.

    Raises:
        Exception: _description_
    """

    def __init__(
        self,
        network_config: Union[CommanderNetworkConfig, nn.Module],
        device: str = "cpu",
        trainning: bool = False,
        learning_rate: float = 2.5e-4,
        eps: float = 1e-8,
        clip_epsilon: float = 0.2,
        value_clip: float = 5.0,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        adv_norm: bool = False,
    ):

        self.device = (
            device if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        )
        print(f"PPOPolciy.device is {self.device}")
        self.trainning = trainning
        self.advantage_normalize = adv_norm
        if isinstance(network_config, CommanderNetworkConfig):
            self._network = ComplexNetwork(network_config)
            # self._network = torch.compile(ComplexNetwork(network_config))
        elif isinstance(network_config, Dict):
            self._network = construct(network_config)
        elif isinstance(network_config, type) and issubclass(network_config, nn.Module):
            self._network = network_config()
        else:
            raise TypeError(f"Unsupport Network Config. [{network_config}] ")

        self._network.to(self.device)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(), lr=learning_rate, eps=eps
        )
        self._loss_fn: PPOLoss = PPOLoss(
            clip_epsilon=clip_epsilon,
            value_clip=value_clip,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        self.max_grad_norm = max_grad_norm

    def train_mode(self):
        self._network.train()

    def inference_mode(self):
        self._network.eval()

    def predict(self, state_input):
        with torch.no_grad():
            outputs = self._network(state_input)
        return outputs

    def learn(self, training_data: Dict[str, Any]):
        inputs_dict = training_data["state_dict"]
        behavior_action_dict = training_data.get("action")
        behavior_logits_dict = training_data.get("logits")
        behavior_mask_dict = training_data.get("decoder_mask")
        behavior_values = training_data.get("value")
        advantages = training_data.get("advantage")

        if self.advantage_normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        target_value = advantages + behavior_values

        with torch.no_grad():
            old_logp_dict_running = self._network.log_probs(
                behavior_logits_dict, behavior_action_dict, behavior_mask_dict
            )
            old_logp = sum(old_logp_dict_running.values())

        predict_output_dict = self._network(
            inputs_dict, behavior_action_dict, training=True
        )
        logits_dict = predict_output_dict["logits"]

        logp_dict = self._network.log_probs(
            logits_dict, behavior_action_dict, behavior_mask_dict
        )
        logp = sum(logp_dict.values())

        # 计算当前策略的熵
        print(logits_dict)
        entropy_dict = self._network.entropy(logits_dict, behavior_mask_dict)
        # entropy = torch.mean(sum(entropy_dict.values()))
        entropy = torch.mean(torch.stack(list(entropy_dict.values())), dim=0)
        print(f"entropy is {entropy}")

        value = predict_output_dict["value"]
        loss, policy_loss, value_loss, entropy_loss, ratio_diff, clipped_fraction = (
            self._loss_fn(
                old_log_prob=old_logp,
                log_prob=logp,
                advantage=advantages,
                old_value=behavior_values,
                value=value,
                target_value=target_value,
                entropy=entropy,
            )
        )
        self._optimizer.zero_grad()
        loss.backward()
        total_norm_before, total_norm_after = check_gradient_clipping(
            self._network, self.max_grad_norm
        )
        self._optimizer.step()
        torch.cuda.empty_cache()  # 释放未使用的显存
        summary_dict = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "ratio_diff": ratio_diff,
            "clipped_fraction": clipped_fraction,
        }

        summary_dict = {k:v.detach().cpu().numpy() for k, v in summary_dict.items()}
        for k, v in summary_dict.items():
            Summary.add_scaler(k, v)
        return summary_dict
