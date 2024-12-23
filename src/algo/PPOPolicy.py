import timeit
import torch
import tree
from hyuRL.src.network.commander import ComplexNetwork
# from hyuRL.src.network.complex import ComplexNetwork
from hyuRL.src.loss.ppo import PPOLoss
from hyuRL.src.memory.buffer import Memory
from hyuRL.src.tools.gpu import auto_move
from typing import Dict, Any


class PPOPolicy:
    def __init__(self, network_config, trainning=False, device="cpu"):
        self.device = device if (device !='cpu' and torch.cuda.is_available()) else 'cpu'
        self.trainning = trainning
        self._network = ComplexNetwork(network_config)
        self._network.to(self.device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss(clip_epsilon=0.2, entropy_coef=0.0)
        self.memory = Memory()

    def train_mode(self):
        self._network.train()

    def inference_mode(self):
        self._network.eval()

    def predict(self, state_input):
        a = timeit.default_timer() * 1000
        with torch.no_grad():
            outputs = self._network(state_input)
        b = timeit.default_timer() * 1000
        return outputs

    def learn(self, trainning_data: Dict[str, Any]):
        inputs_dict = trainning_data["state_dict"]
        behavior_action_dict = trainning_data.get("action")
        behavior_logits_dict = trainning_data.get("logits")
        behavior_mask_dict = trainning_data.get("action_mask")
        behavior_values = trainning_data.get("value")
        advantages = trainning_data.get("advantage")
        target_value = advantages + behavior_values


        with torch.no_grad():
            old_logp_dict_running = self._network.log_probs(
                behavior_logits_dict, behavior_action_dict, behavior_mask_dict
            )
            old_logp = sum(old_logp_dict_running.values())

        eplased_times = []
        import time

        for epoch in range(10):
            btime = time.time()
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
            loss, policy_loss, value_loss, entropy_loss, ratio, clipped_mask = (
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
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
            self._optimizer.step()
            eplased_times.append(round(time.time() * 1000 - btime * 1000, 1))
        return {
            "loss": loss.detach(),
            # "policy_loss": policy_loss,
            # "value_loss": value_loss,
            # "entropy": entropy,
            # "ratio_diff": ratio,
        }
