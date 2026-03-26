import gymnasium as gym
import numpy as np
from hyurl.flow.drill_plugin.api.api import ObsData


class CartPoleEnv:
    def __init__(self, env_id=None, extra_info=None):
        self.env = gym.make("CartPole-v1")
        self.agent_names = ["cpdemo"]
        self.env_id = env_id
        self.extra_info = extra_info

    def reset(self):
        self.total_reward = 0
        raw_obs, _ = self.env.reset()
        return {
            agent_name: ObsData(
                obs=raw_obs,
                extra_info_dict={
                    "reward": 1.0,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }

    def step(self, command_dict):
        action = command_dict[self.agent_names[0]]
        raw_obs, reward, terminated, truncated, info = self.env.step(action=np.array(action['action']).item())
        self.total_reward += reward
        done = terminated or truncated
        
        return {
            agent_name: ObsData(
                obs=raw_obs,
                extra_info_dict={
                    "reward": reward,
                    "episode_reward": self.total_reward,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }, done
