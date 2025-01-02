from hyuRL.src.feature.feature import *
from hyuRL.src.feature.feature_set import *
from hyuRL.src.api.net.net import *
from hyuRL.src.flow.drill_plugin.interface.builder import ExBuilder
from hyuRL.src.algo.PPOPolicy import PPOPolicy
from hyuRL.example.atari.implement import GymEnv
from hyuRL.example.atari.implement import PipelineImplement
from drill.pipeline import AgentPipeline, HandlerSpecies

network_cfg = CommanderNetworkConfig(
    encoders=[
        SpatialEncoderConfig(
            channel_num = 4,
            down_samples = [
                (32, 8, 4, 0),
                (64, 4, 2, 0),
                (64, 3, 1, 0),
            ],
            res_block_num = 0,
            feature_set=SpatialFeatureSet(
                name="common",
                shape=[84, 84],
                feature_dict={"raw": VectorFeature(4)}
            )
        ),
    ],
    decoders=[
        CategoricalDecoderConfig(name="meta_action", n=9),
    ],
)

model_config = {
    "atari_model": {
        "class": PPOPolicy,  # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network_config": network_cfg,  # 神经网络结构
            "device": "cuda",
        },
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_list = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders]

pipeline = {
    "atari_pipeline": {
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
            # "batch_config": {  # advantage
            #     "gamma": 0.99,
            #     "lamb": 0.95,
            # },
        },
    }
}

agents = {"atari_agent": {"model": "atari_model", "pipeline": "atari_pipeline"}}


env = {
    "class": GymEnv,
    "params": {
        "atari_info": {
            "atari_env_args" : {
                "id": "ALE/BeamRider-v5",
            },
            "image_dim": 84,
            "agent_names": list(agents.keys())
        }

    },
}

builder = ExBuilder(agents, model_config, env, pipeline)
