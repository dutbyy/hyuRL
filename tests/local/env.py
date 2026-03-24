import gymnasium as gym
import numpy as np
import torch
from typing import Tuple, Dict, Any


class CartPoleEnv:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.env.reset()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def get_observation_dict(self, observation: np.ndarray) -> Dict[str, torch.Tensor]:
        return {"observation": torch.from_numpy(observation).float().unsqueeze(0)}
