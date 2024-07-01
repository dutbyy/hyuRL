import logging
import timeit
import numpy as np
import torch
from torch import optim
import gym
import tree

from src.network.complex import ComplexNetwork
from src.loss.ppo import PPOLoss
from src.tools.gae import calculate_gae
from src.memory.buffer import Fragment, Memory
from src.tools.gpu import auto_move
from src.tools.common import timer_decorator


class PPOPolicy:
    def __init__(self, policy_config, trainning=False, device="cpu"):
        self._network = ComplexNetwork(policy_config)
        # if not trainning:
        # self._network = torch.compile(self._network)
        self.device = device
        self._network.to(self.device)
        self._optimizer = optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss(clip_epsilon=0.2, entropy_coef=0.0)
        self.memory = Memory()

    @timer_decorator
    def inference(self, state_input):
        # inputs = {"common_encoder": torch.Tensor(state)}
        a = timeit.default_timer() * 1000
        inputs = auto_move(state_input, self.device)
        # print("move input")
        with torch.no_grad():
            # print('before network cal')
            outputs = self._network(inputs)

            # print(f"inference 耗时 {b-a}")
            # outputs = self._network(inputs)
        # print(outputs)
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
        old_logp_dict_running = trainning_data.get("log_probs")
        old_logp = sum(
            old_logp_dict_running.values()
        )  # 在动作维度合并logp (相当于动作概率连乘)
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
            # logging.info(f"loss: {loss}, ratio: {ratio}")
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
            self._optimizer.step()
        return


def sample(pipe, agent):
    from memory.buffer import Fragment

    fragment = Fragment
    env = gym.make("CartPole-v1", max_episode_steps=500)
    state, _ = env.reset()
    total_reward = 0
    episode_info = []
    while True:
        outputs = agent.inference(state)
        nstate, reward, done, truncted, info = env.step(
            outputs["action"]["action"].item()
        )
        reward *= 1
        total_reward += 1
        done = done or truncted
        action_mask = {"action": torch.tensor(1)}
        logp = agent._network.log_probs(
            outputs["logits"], outputs["action"], action_mask
        )
        episode_info.append(
            {"common_encoder": torch.Tensor(state)},
            outputs["action"],
            reward,
            1 if done else 0,
            logp,
            action_mask,
            outputs["value"],
            outputs["logits"],
        )
        fragment.store(
            {"common_encoder": torch.Tensor(state)},
            outputs["action"],
            reward,
            1 if done else 0,
            logp,
            action_mask,
            outputs["value"],
            outputs["logits"],
        )

        if len(fragment) >= 128:
            pipe
            fragment = Fragment()
        if done:
            total_rewards.append(total_reward)
            break
        state = nstate


if __name__ == "__main__":
    from network.encoder.common import CommonEncoder
    from network.decoder.categorical import CategoricalDecoder
    from network.app_value import ValueApproximator
    from network.aggregator.dense import DenseAggregator

    env = gym.make("CartPole-v1", max_episode_steps=200)

    encoder_config = {
        "common_encoder": {
            "class": CommonEncoder,
            "params": {
                "in_features": 4,
                "hidden_layer_sizes": [256, 128],
            },
        }
    }
    decoder_config = {
        "action": {
            "class": CategoricalDecoder,
            "params": {
                "n": 2,
                "hidden_layer_sizes": [256, 128],
                # "temperature": 0.5,
            },
        }
    }
    aggregator_config = {
        "class": DenseAggregator,
        "params": {
            "in_features": 128,
            "hidden_layer_sizes": [256, 128],
            "output_size": 256,
        },
    }
    value_config = {
        "class": ValueApproximator,
        "params": {
            "hidden_layer_sizes": [256],
        },
    }

    agent = PPOPolicy(encoder_config, aggregator_config, decoder_config, value_config)
    EpisodeNum = 5000

    avg_reward = 0
    train_step = 0
    total_rewards = []
    import timeit

    for episode in range(EpisodeNum):
        inputs_dict = {}
        state, _ = env.reset()
        total_reward = 0
        # print("Reset The Environment")
        while True:
            outputs = agent.inference(state)
            nstate, reward, done, truncted, info = env.step(
                outputs["action"]["action"].item()
            )
            reward *= 1
            reward -= nstate[0] * 0.1
            reward -= nstate[1] * 0.1
            reward -= nstate[2] * 0.3
            reward -= nstate[3] * 0.3
            total_reward += 1
            done = done or truncted

            action_mask = {"action": torch.tensor(1)}
            logp = agent._network.log_probs(
                outputs["logits"], outputs["action"], action_mask
            )
            agent.memory.store(
                {"common_encoder": torch.Tensor(state)},
                outputs["action"],
                reward,
                1 if done else 0,
                logp,
                action_mask,
                outputs["value"],
                outputs["logits"],
            )

            if agent.memory.size >= 1024:
                print(
                    f"{train_step:4}: average reward is {int(sum(total_rewards)/len(total_rewards))}",
                    end="\t",
                )
                start_time = timeit.default_timer()
                agent.train()
                end_time = timeit.default_timer()
                print(f"train eplased time : {end_time - start_time:.2}")
                train_step += 1
                agent.memory.reset()
                total_rewards = []
                break
            if done:
                total_rewards.append(total_reward)
                break
            state = nstate

            # agent.clean_data()
