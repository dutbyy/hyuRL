import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from drill.pipeline.interface import ObsData, ActionData
from drill import summary
import logging
from .atari_wrappers import wrap_deepmind
from stable_baselines3.common.env_util import make_atari_env

def getLogger(env_id):
    log_name = env_id if isinstance(env_id, str) else f"env-{env_id}"
    logger = logging.getLogger(f"env-{env_id}")
    logger.setLevel(20)
    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    try:
        import os
        os.system("mkdir -p /job/logs/user_log/")
        handler = logging.FileHandler(f"/job/logs/user_log/{log_name}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except:
        pass
    return logger


class GymEnv:
    def __init__(self, env_id, atari_info, extra_info):
        atari_env_args = atari_info.get("atari_env_args")
        dim = atari_info.get("image_dim", 64)
        # self.env = wrap_deepmind(gym.make(**atari_env_args), dim=dim, framestack=False)
        # self.env = wrap_deepmind(gym.make(**atari_env_args), dim=dim)
        self.env = make_atari_env('ALE/BeamRider-v5', n_envs=1, seed=0)


        self.agent_names = atari_info.get("agent_names", [])
        self.logger = getLogger(env_id)
        self.hist_rewards = []

    def reset(self):
        self.total_reward = 0
        self.step_num = 0
        raw_obs = self.env.reset()
        return {
            agent_name: ObsData(
                obs = raw_obs.squeeze(0) / 255.0,
                extra_info_dict = {
                    "reward": 0.0,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }

    def step(self, command_dict):
        mean = lambda x : sum(x)/len(x)
        self.step_num += 1
        action = command_dict[self.agent_names[0]]
        raw_obs, reward, terminated, info = self.env.step(actions=np.expand_dims(action['meta_action'], 0))
        truncated = terminated
        self.total_reward += reward.item()
        if terminated or truncated:
            summary.average("episode_reward", self.total_reward)
            summary.average("episode_step", self.step_num)
            self.hist_rewards.append(self.total_reward)
            if len(self.hist_rewards) >= 10:
                self.logger.info(f"10 episode Over, Total reward is {mean(self.hist_rewards):.2f}")
                self.hist_rewards.clear()
        return {
                agent_name: ObsData(
                    obs = raw_obs.squeeze(0) / 255.0,
                    extra_info_dict= {
                        "reward": reward.item(),
                        "episode_reward": info[0].get('episode', {}).get('r', 0.0)
                    },
                    agent_name=agent_name
                )
                for agent_name in self.agent_names
            }, terminated or truncated

class PipelineImplement:
    @staticmethod
    def feature_handler(obs_data:ObsData, history):
        return {
            "common": {
                "raw": obs_data.obs
            }
        }

    @staticmethod
    def reward_handler(obs_data:ObsData, history):
        return obs_data.extra_info_dict.get("reward", 1.0)

    @staticmethod
    def action_handler(action_data:ActionData, history):
        action_data.action_mask = {k: np.ones(1) for k in action_data.action}
        return action_data
