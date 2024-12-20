

from hyuRL.src.feature.feature import *
from hyuRL.src.feature.feature_set import *
from hyuRL.src.api.net.net import *
from hyuRL.src.network.commander import ComplexNetwork
import torch
from hyuRL.src.flow.drill_plugin.interface.builder import ExBuilder


network_cfg = CommanderNetworkConfig(
    encoders = [
        CommonEncoderConfig(
            feature_set = CommonFeatureSet(
                name = "common",
                feature_dict = {
                    "raw": VectorFeature(4)
                }
            )
        ),
        # EntityEncoderConfig(
        #     feature_set = EntityFeatureSet(
        #         name = "enemies",
        #         max_length = 10,
        #         feature_dict = {
        #             "space": VectorFeature(16)
        #         }
        #     )
        # ),
        # SpatialEncoderConfig(
        #     feature_set = SpatialFeatureSet(
        #         name = "heights",
        #         shape = [16, 16],
        #         feature_dict = {
        #             "space": VectorFeature(10)
        #         }
        #     )
        # ),
    ],
    decoders = [
        CategoricalDecoderConfig(name="meta_action", n=2),
        # GaussianDecoderConfig(name="direction", n=1),
        # SingleSelectiveDecoderConfig(name="target", source_encoder_name="encoder_enemies"),
    ]
) 
    

from src.algo.PPOPolicy import PPOPolicy
model_config = {
    "cp_model": {
        "class": PPOPolicy,     # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network_config": network_cfg,         # 神经网络结构
            # "learning_rate": 2e-4,      # 学习率
            # "clip_param": 0.3,          # 为了训练稳定，限制新旧 policy network 的差距
            # "vf_clip_param": 10.0,      # 为了训练稳定，限制 value network 的差距
            # "vf_loss_coef": 1.0,        # value loss 的 scale factor（影响因子）, 为 1 则不 scale
            # "entropy_coef": 0.1,        # entropy loss 的 scale factor（影响因子）, 为 1 则不 scale
        },
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_list = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders]

from drill.pipeline import AgentPipeline, HandlerSpecies
from hyuRL.example.cartpole.implement import PipelineImplement
pipeline = {
    "cp_pipeline": {
        "class": AgentPipeline,
        "params": {
            "handler_dict": {
                HandlerSpecies.FEATURE: (
                    PipelineImplement.feature_handler,
                    feature_list,
                ),
                HandlerSpecies.REWARD: PipelineImplement.reward_handler,
                HandlerSpecies.ACTION: PipelineImplement.action_handler,
            },
            "batch_config": {   # advantage
                "gamma": 0.99,
                "lamb": 0.95,
            }
        }
    }
}

agents = {
    "cpdemo": {
        "model": "cp_model",
        "pipeline": "cp_pipeline"
    }
}

import gym
from drill.pipeline.interface import ObsData, ActionData

class CartpoleEnv:
    def __init__(self, env_id, extra_info):
        self.env = gym.make("CartPole-v0")
        self.agent_names = ["cpdemo"]

    def reset(self): 
        self.total_reward = 0
        
        raw_obs, _ = self.env.reset()
        return {
            agent_name: ObsData(
                obs = raw_obs,
                extra_info_dict={
                    "reward": 1.0,
                },
                agent_name=agent_name
            )
            for agent_name in self.agent_names
        }
    
    def step(self, command_dict):
        print(f"command : {command_dict}")
        action = command_dict[self.agent_names[0]]
        raw_obs, reward, truncted, done, extra_info = self.env.step(action=np.array(action['meta_action']))
        self.total_reward += 1
        if done or truncted:
            print(f"Episode Over, Total reward is {self.total_reward}")
        return {
                agent_name: ObsData(
                    obs = raw_obs,
                    extra_info_dict= {
                        "reward": 1.0,
                    },
                    agent_name=agent_name
                )
                for agent_name in self.agent_names
            }, truncted or done

env = {
    "class" : CartpoleEnv,
    "params": {
        # "id": "CartPole-v0",
        # "max_episode_steps": 500,
    }
}

builder = ExBuilder(agents, model_config, env, pipeline)
print(builder)