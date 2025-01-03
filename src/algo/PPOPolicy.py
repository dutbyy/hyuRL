import timeit
import torch
import tree
from hyuRL.src.network.commander import ComplexNetwork
# from hyuRL.src.network.complex import ComplexNetwork
from hyuRL.src.loss.ppo import PPOLoss
from hyuRL.src.memory.buffer import Memory
from typing import Dict, Any
from torch import nn

def check_gradient_clipping(model, max_grad_norm):
    # 计算梯度范数
    total_norm_before = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]), 2.0)

    # 执行梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 计算裁剪后的梯度范数
    total_norm_after = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]), 2.0)

    # 返回裁剪信息
    return total_norm_before, total_norm_after


class PPOPolicy:
    def __init__(self, network_config, trainning=False, device="cpu"):
        self.device = device if (device !='cpu' and torch.cuda.is_available()) else 'cpu'
        print(f"PPOPolciy.device is {self.device}")
        self.trainning = trainning
        if issubclass(network_config, nn.Module):
            self._network = network_config()
        else:
            self._network = ComplexNetwork(network_config)
        self._network.to(self.device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=3e-4)
        self._loss_fn = PPOLoss(clip_epsilon=0.2, entropy_coef=0.0)
        self.max_grad_norm = 1.0
        self.memory = Memory()

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
        entropy_dict = self._network.entropy(logits_dict, behavior_mask_dict)
        entropy = torch.mean(sum(entropy_dict.values()))
        value = predict_output_dict["value"]
        loss, policy_loss, value_loss, entropy_loss, ratio, clipped_fraction = (
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
        # print(f"loss: {loss}  policy_loss: {policy_loss} value_loss: {value_loss} ")
        loss.backward()
        total_norm_before, total_norm_after = check_gradient_clipping(self._network, self.max_grad_norm)
        # if total_norm_before > self.max_grad_norm:
        #     print(f"发生了梯度裁剪: 裁剪前范数 = {total_norm_before}, 裁剪后范数 = {total_norm_after}")
        # else:
        #     print(f"未发生梯度裁剪: 梯度范数 = {total_norm_before}")

        # torch.nn.utils.clip_grad_norm_(self._network.parameters(), 0.5)
        self._optimizer.step()
        return {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "clipped_fraction": clipped_fraction.detach(),
            # "ratio_diff": ratio,
        }
