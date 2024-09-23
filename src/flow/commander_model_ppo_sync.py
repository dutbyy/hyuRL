from typing import Any, Dict
from drill.keys import ACTION, ADVANTAGE, DECODER_MASK, LOGITS, VALUE
from drill.model.model import Model
from drill.utils import get_hvd, tf_normalize_advantage
import timeit
import torch
import tree
from src.network import ComplexNetwork
from src.loss.ppo import PPOLoss
import numpy as np
import traceback 

class PPOPolicyModel(Model):
    def __init__(self,
                 network: ComplexNetwork,
                 learning_rate: float = 5e-4,
                 clip_param: float = 0.1,
                 vf_clip_param: int = 10,
                 vf_loss_coef: float = 1,
                 entropy_coef: float = 0.02,
                 **kwargs):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        self._loss_fn = PPOLoss(clip_epsilon=clip_param, entropy_coef=entropy_coef)

        hvd = get_hvd("pytorch")
        hvd.init()
        
    @property
    def network(self):
        return self._network

    def predict(self, state_dict: dict) -> dict:
        with torch.no_grad():
            outputs = self._network(state_dict)
        return outputs
            
    def learn(self, inputs_dict: Dict[str, Any], behavior_info_dict: Dict[str,Any], episode=10) -> Dict[str, Any]:
        import time
        eplased_times = []

        atime = time.time() * 1000

        behavior_action_dict = behavior_info_dict.get("action")
        behavior_mask_dict = behavior_info_dict.get("decoder_mask")
        behavior_values = behavior_info_dict.get("value")
        behavior_logits_dict = behavior_info_dict.get("logits")
        advantages = behavior_info_dict.get("advantage")
        logits = behavior_info_dict.get("advantage")
        target_value = advantages + behavior_values

#         old_logp_dict = behavior_info_dict.get("log_probs")
#         old_logp = sum(
#             old_logp_dict.values()
#         ) 
#         self.logger.info(f"input dict is : {inputs_dict}")
#         self.logger.info(f"behavior_action_dict is : {behavior_action_dict}")
        with torch.no_grad():
            old_logp_dict_running = self._network.log_probs(
                behavior_logits_dict, behavior_action_dict, behavior_mask_dict
            )
            old_logp = sum(old_logp_dict_running.values())
        btime = time.time() * 1000
        eplased_times.append(round(btime-atime, 1))
        summary = {}
        for epoch in range(episode):
            atime = time.time() * 1000
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
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
            self._optimizer.step()
            
            btime = time.time() * 1000
            eplased_times.append(round(btime-atime, 1))
        self.logger.info(f"learner eplased_time: {eplased_times}")
        summary = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            # "clip_prob": clip_prob,
            # "ratio_diff": ratio_diff,
            # "advantage": mean_advantage,
            # "kl": kl,
        }
        return summary
