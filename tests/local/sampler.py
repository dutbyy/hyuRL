import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from .env import CartPoleEnv


class Transition:
    def __init__(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        value: float,
        logits: Dict[str, torch.Tensor],
    ):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.done = done
        self.value = value
        self.logits = logits


class Sampler:
    def __init__(self, model, env: CartPoleEnv, device: str = "cpu"):
        self.model = model
        self.env = env
        self.device = device
        self.transitions: List[Transition] = []
        self.episode_rewards: List[float] = []

    def collect(self, num_episodes: int = 1) -> List[Transition]:
        self.transitions = []
        self.episode_rewards = []

        for episode in range(num_episodes):
            observation, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                obs_dict = self.env.get_observation_dict(observation)
                with torch.no_grad():
                    output = self.model(obs_dict, training=False)

                action_logits = output["logits"]["action"]
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                transition = Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                    value=output["value"].item(),
                    logits={"action": action_logits.squeeze(0)},
                )
                self.transitions.append(transition)

                observation = next_observation

            self.episode_rewards.append(episode_reward)

        return self.transitions

    def get_batch(self, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        if len(self.transitions) < batch_size:
            batch_size = len(self.transitions)

        indices = np.random.choice(len(self.transitions), batch_size, replace=False)

        observations = np.array([self.transitions[i].observation for i in indices])
        actions = np.array([self.transitions[i].action for i in indices])
        rewards = np.array([self.transitions[i].reward for i in indices])
        next_observations = np.array([self.transitions[i].next_observation for i in indices])
        dones = np.array([self.transitions[i].done for i in indices], dtype=np.float32)
        values = np.array([self.transitions[i].value for i in indices])

        return {
            "observations": torch.from_numpy(observations).float(),
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
            "next_observations": torch.from_numpy(next_observations).float(),
            "dones": torch.from_numpy(dones).float(),
            "values": torch.from_numpy(values).float(),
        }

    def get_statistics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "num_episodes": len(self.episode_rewards),
        }
