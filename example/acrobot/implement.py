
import gymnasium
import numpy as np
from drill.pipeline.interface import ObsData, ActionData
from drill import summary
import logging

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

class AcrobotEnv:
    def __init__(self, env_id, extra_info):
        self.env = gymnasium.make("Acrobot-v1")
        self.agent_names = ["acdemo"]
        self.logger = getLogger(env_id)
        self.hist_rewards = []

    def reset(self):
        self.total_reward = 0

        raw_obs, _ = self.env.reset()
        return {
            agent_name: ObsData(
                obs = raw_obs,
                extra_info_dict = {
                    "reward": 0.0,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }

    def step(self, command_dict):
        action = command_dict[self.agent_names[0]]
        # print(action)
        raw_obs, reward, truncted, done, extra_info = self.env.step(action=np.array(action['meta_action']).item())
        self.total_reward += reward
        if done or truncted:
            summary.average("episode_reward", self.total_reward)
            self.hist_rewards.append(self.total_reward)
            if len(self.hist_rewards) >= 10:
                mean = lambda x : sum(x)/len(x)
                import os
                import threading
                self.logger.info(f"10 episode Over, Total reward is {mean(self.hist_rewards):.2f}")
                # print(f"10 episode Over, Total reward is {mean(self.hist_rewards):.2f}")
                self.hist_rewards.clear()

        return {
                agent_name: ObsData(
                    obs = raw_obs,
                    extra_info_dict= {
                        "reward": reward,
                        "episode_reward": self.total_reward,
                    },
                    agent_name=agent_name
                )
                for agent_name in self.agent_names
            }, truncted or done

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
