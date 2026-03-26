from hyurl.feature.feature import *
from hyurl.feature.feature_set import *
from hyurl.api.net.net import *
from hyurl.flow.drill_plugin.interface.builder import ExBuilder
from hyurl.algo.PPOPolicyMinibatch import PPOPolicy
from hyurl.api.agent.agent_pipeline import Agent
from .implement import CartpoleEnv
from .implement import PipelineImplement
# from .implement import CartpoleAgent

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
            "device": "cpu",
            "epoch_num": 10,
            "minibatch_split": 4,
            "entropy_coef": 0.0,
            "eps": 1e-5,
            "max_grad_norm": 1000,
        },
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_sets = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders]

pipeline = {
    "cp_pipeline": {
        "class": Agent,
        "params": {
            "feature_sets": feature_sets,
            "pipeline": PipelineImplement(),
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