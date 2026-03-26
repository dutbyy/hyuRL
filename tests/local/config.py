from hyurl.feature.feature import *
from hyurl.feature.feature_set import *
from hyurl.api.net.net import *
from hyurl.flow.drill_plugin.interface.builder import ExBuilder
from hyurl.algo.PPOPolicyMinibatch import PPOPolicy
from hyurl.api.agent.agent_pipeline import Agent
from tests.local.env import CartPoleEnv
from tests.local.implement import PipelineImplement

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
        CategoricalDecoderConfig(name="action", n=2, hidden_layer_sizes=[]),
    ],
    aggregator=DenseAggregatorConfig(hidden_layer_sizes=[]),
    value=ValueApproximatorConfig(hidden_layer_sizes=[]),
)

model_config = {
    "cp_model": {
        "class": PPOPolicy,
        "params": {
            "network_config": network_cfg,
            "device": "cpu",
            "learning_rate": 3e-4,
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "adv_norm": False,
        },
        "save": {
            "interval": 100,
            "mode": "pth",
            "path": "./model",
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
    "class": CartPoleEnv,
    "params": {},
}

builder = ExBuilder(agents, model_config, env, pipeline)
