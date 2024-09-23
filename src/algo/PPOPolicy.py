import timeit
import torch
import tree
from src.network.complex import ComplexNetwork
from src.loss.ppo import PPOLoss
from src.memory.buffer import Memory
from src.tools.gpu import auto_move
from typing import Dict, Any


class PPOPolicy:
    def __init__(self, network_config, trainning=False, device="cpu"):
        self.device = device if (device !='cpu' and torch.cuda.is_available()) else 'cpu'
        self.trainning = trainning
        self._network = ComplexNetwork(network_config)
        self._network.to(self.device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss(clip_epsilon=0.1, entropy_coef=0.0)
        self.memory = Memory()

    def train_mode(self):
        self._network.train()

    def inference_mode(self):
        self._network.eval()

    def inference(self, state_input):
        a = timeit.default_timer() * 1000

        state_input = tree.map_structure(
            lambda x: torch.from_numpy(x).cuda() if self.device == 'cuda' else torch.from_numpy(x),
            state_input,
        )

        with torch.no_grad():
            outputs = self._network(state_input)

        outputs = tree.map_structure(
            lambda x: x.numpy(),
            outputs,
        )

        b = timeit.default_timer() * 1000
        eplased_time = b - a
        return outputs, eplased_time

    def train(self, trainning_data: Dict[str, Any]):
        trainning_data = tree.map_structure(
            lambda x: torch.from_numpy(x).cuda() if self.device == 'cuda' else torch.from_numpy(x),
            trainning_data,
        )
        
        inputs_dict = trainning_data["states"]
        behavior_action_dict = trainning_data.get("actions")
        behavior_logits_dict = trainning_data.get("logits")
        behavior_mask_dict = trainning_data.get("masks")
        behavior_values = trainning_data.get("values")
        advantages = trainning_data.get("advantages")
        target_value = advantages + behavior_values
        # old_logp_dict_jilu = trainning_data.get("log_probs")
        # old_logp = sum(
        #     old_logp_dict_jilu.values()
        # )  # 在动作维度合并logp (相当于动作概率连乘)

        with torch.no_grad():
            old_logp_dict_running = self._network.log_probs(
                behavior_logits_dict, behavior_action_dict, behavior_mask_dict
            )
            old_logp = sum(old_logp_dict_running.values())

        eplased_times = []
        import time

        for epoch in range(20):
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
        return eplased_times
