from hyuRL.src.feature.feature import *
from hyuRL.src.feature.feature_set import *
from hyuRL.src.api.net.net import *
from hyuRL.src.flow.drill_plugin.interface.builder import ExBuilder
# from hyuRL.src.algo.PPOPolicy import PPOPolicy
from hyuRL.src.algo.PPOPolicyMinibatch import PPOPolicy
from hyuRL.example.cartpole.implement import CartpoleEnv
from hyuRL.example.cartpole.implement import PipelineImplement
from drill.pipeline import AgentPipeline, HandlerSpecies

network_cfg = CommanderNetworkConfig(
    encoders=[
        CommonEncoderConfig(
            hidden_layer_sizes=[],
            feature_set=CommonFeatureSet(
                name="common", feature_dict={"raw": VectorFeature(4)}
            ),
        ),
    ],
    decoders=[
        CategoricalDecoderConfig(name="meta_action", n=2, hidden_layer_sizes=[]),
    ],
    aggregator=DenseAggregatorConfig(hidden_layer_sizes=[]),
    value=ValueApproximatorConfig(hidden_layer_sizes=[]),
)

model_config = {
    "cp_model": {
        "class": PPOPolicy,  # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network_config": network_cfg,  # 神经网络结构
            "device": "cuda",
            "epoch_num": 10,
            "minibatch_split": 4,
            "entropy_coef": 0.0,
            "eps": 1e-5,
        },
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_list = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders]

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
            "batch_config": {  # advantage
                "gamma": 0.99,
                "lamb": 0.95,
            },
        },
    }
}

agents = {"cpdemo": {"model": "cp_model", "pipeline": "cp_pipeline"}}


env = {
    "class": CartpoleEnv,
    "params": {
    },
}

builder = ExBuilder(agents, model_config, env, pipeline)
print(builder)
