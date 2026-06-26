from hyurl.feature.feature import *
from hyurl.feature.feature_set import *
from hyurl.api.net.net import *
from hyurl.api.agent.agent_pipeline import Agent
from hyurl.flow.drill_plugin.interface.builder import ExBuilder
from hyurl.algo.PPOPolicy import PPOPolicy

from example.acrobot.implement import AcrobotEnv
from example.acrobot.implement import PipelineImplement

network_cfg = CommanderNetworkConfig(
    encoders=[
        CommonEncoderConfig(
            hidden_layer_sizes=[],
            feature_set=CommonFeatureSet(
                name="common", feature_dict={"raw": VectorFeature(6)}
            ),
        ),
    ],
    decoders=[
        CategoricalDecoderConfig(name="meta_action", n=3, hidden_layer_sizes=[]),
    ],
    aggregator=DenseAggregatorConfig(hidden_layer_sizes=[]),
    value=ValueApproximatorConfig(hidden_layer_sizes=[]),
)

model_config = {
    "acrobot_model": {
        "class": PPOPolicy,  # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network_config": network_cfg,  # 神经网络结构
            "device": "cuda",
            "learning_rate": 2.5e-4,
            "clip_epsilon": 0.2,
            "eps": 1e-9,
            "device": "cuda",
            "max_grad_norm": 0.5,
            # "adv_norm": True,
        },
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_sets = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders]

pipeline = {
    "ac_pipeline": {
        "class": Agent,
        "params": {
            "feature_sets": feature_sets,
            "pipeline": PipelineImplement(),
        },
    }
}

agents = {"acrobot-demo": {"model": "acrobot_model", "pipeline": "ac_pipeline"}}


env = {
    "class": AcrobotEnv,
    "params": {
    },
}

builder = ExBuilder(agents, model_config, env, pipeline)
from pprint import pprint
pprint(builder)
