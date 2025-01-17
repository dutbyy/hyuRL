import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing
import numpy as np
from drill.pipeline.interface import ObsData, ActionData
from drill import summary
import logging
from .atari_wrappers_ray import wrap_deepmind
from .atari_wrappers_sb3 import AtariWrapper

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
        self.env_id = env_id
        atari_env_args = atari_info.get("atari_env_args")
        dim = atari_info.get("image_dim", 84)
        self.env = wrap_deepmind(gym.make(**atari_env_args), dim=dim, noframeskip=True)
        # self.env = AtariWrapper(env=gym.make(**atari_env_args), dim=dim, frame_stack=1)
        # self.env = AtariPreprocessing(
        #     env=gym.make(**atari_env_args),
        #     screen_size=dim,
        #     scale_obs=True,
        #     grayscale_newaxis=True,
        # )
        self.agent_names = atari_info.get("agent_names", [])
        self.logger = getLogger(env_id)

    def reset(self):
        self.total_reward = 0
        self.step_num = 0
        raw_obs, info = self.env.reset()
        return {
            agent_name: ObsData(
                obs = raw_obs,
                extra_info_dict = {
                    "reward": 0.0,
                    "done": False,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }

    def step(self, command_dict):
        self.step_num += 1
        action = command_dict[self.agent_names[0]]
        raw_obs, reward, terminated, truncated, info = self.env.step(action=action['meta_action'])
        self.total_reward += reward
        if terminated:
            self.logger.info(f"episode Over, clipped reward is {self.total_reward}")
            summary.average("episode_reward", self.total_reward)
            self.logger.info(f"episode Over, natural reward is {info['nature_episode_reward']}")
            summary.average("episode_natural_reward", info['nature_episode_reward'])
        return {
            agent_name: ObsData(
                obs = raw_obs,
                extra_info_dict={
                    "reward": reward,
                    "episode_reward": info['nature_episode_reward'],
                    "lives": info["lives"],
                },
                agent_name=agent_name,
            )
            for agent_name in self.agent_names
        }, terminated

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
