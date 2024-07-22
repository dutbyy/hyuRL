import timeit
import torch
import gym
import tree
import logging
import numpy as np
from src.network.complex import ComplexNetwork
from src.loss.ppo import PPOLoss
from src.memory.buffer import Fragment, Memory
from src.tools.gpu import auto_move

class PPOPolicy:
    def __init__(self, policy_config, trainning=False, device="cpu"):
        self.device = device
        self.policy_config = policy_config
        self.trainning = trainning
        
        self._network = ComplexNetwork(policy_config)
        self._network.to(self.device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss(clip_epsilon=0.2, entropy_coef=0.0)
        self.memory = Memory()
        
    def train_mode(self):
        self._network.train()
        
    def inference_mode(self):
        self._network.eval()

    def inference(self, state_input):
        a = timeit.default_timer() * 1000
        inputs = auto_move(state_input, self.device)
        with torch.no_grad():
            outputs = self._network(inputs)
        outputs = tree.map_structure(
            lambda x: x.to("cpu").numpy() if isinstance(x, torch.Tensor) else None,
            outputs,
        )
        b = timeit.default_timer() * 1000
        eplased_time = b - a
        return outputs, eplased_time

    def train(self, trainning_data):
        # trainning_data  =  self.memory.get_batch(1024)
        trainning_data = auto_move(trainning_data, self.device)
        inputs_dict = trainning_data["states"]
        behavior_action_dict = trainning_data.get("actions")
        behavior_logits_dict = trainning_data.get("logits")
        behavior_mask_dict = trainning_data.get("masks")
        behavior_values = trainning_data.get("values")
        advantages = trainning_data.get("advantages")
        target_value = advantages + behavior_values
        # old_logp_dict_running = trainning_data.get("log_probs")
        # old_logp = sum(
        #     old_logp_dict_running.values()
        # )  # 在动作维度合并logp (相当于动作概率连乘)
        
        with torch.no_grad():
            old_logp_dict_running = self._network.log_probs(
                behavior_logits_dict, behavior_action_dict, behavior_mask_dict
            )
            old_logp = sum(old_logp_dict_running.values())
            
            
        for epoch in range(30):
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
            # kl_dict = self._network.kl(logits_dict, behavior_logits_dict, behavior_mask_dict )
            # kl = sum(kl_dict.values()).mean()
            # print("kl value", kl)
            # print("clipped mask", clipped_mask.mean())
            self._optimizer.zero_grad()
            loss.backward()
            # print(f"loss: {loss}, ratio: {torch.mean(torch.abs(ratio-1)).cpu().detach().numpy()}")
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
            self._optimizer.step()
        return
