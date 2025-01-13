from hyuRL.src.feature.feature import *
from hyuRL.src.feature.feature_set import *
from hyuRL.src.api.net.net import *
from hyuRL.src.flow.drill_plugin.interface.builder import ExBuilder
from hyuRL.src.algo.PPOPolicy import PPOPolicy
from hyuRL.example.minigame.implement import MinigameEnv
from hyuRL.example.minigame.implement import PipelineImplement
from drill.pipeline import AgentPipeline, HandlerSpecies

# marine 实体特征
marine_feature_set = EntityFeatureSet(
    "marine",
    feature_dict={
        "health": PlainFeature(),
        "x_loc": PlainFeature(),
        "y_loc": PlainFeature(),
    },
    max_length=9,
)

# enemy 实体特征，包括 Baneling, Zergling
enemy_feature_set = EntityFeatureSet(
    "enemy",
    feature_dict={
        "type": OnehotFeature(2),
        "health": PlainFeature(),
        "x_loc": PlainFeature(),
        "y_loc": PlainFeature(),
    },
    max_length=10,
)

# enemy mask
enemy_mask_feature_set = CommonFeatureSet("enemy_mask", feature_dict={"attack_target": VectorFeature(10)})

# common 环境的统计信息
common_feature_set = CommonFeatureSet(
    name="common",
    feature_dict={
        "marine_alive_number": PlainFeature(),
        "baneling_alive_number": PlainFeature(),
        "zergling_alive_number": PlainFeature(),
    },
)

network_cfg = CommanderNetworkConfig(
    encoders=[
        EntityEncoderConfig(
            hidden_layer_sizes=[256, 128],
            feature_set=marine_feature_set,
        ),
        EntityEncoderConfig(
            hidden_layer_sizes=[256, 128],
            feature_set=enemy_feature_set,
        ),
        CommonEncoderConfig(
            hidden_layer_sizes=[256, 128],
            feature_set=common_feature_set,
        ),
    ],
    decoders=[
        CategoricalDecoderConfig(name="meta", n=2, hidden_layer_sizes=[]),
        UnorderedMultiSelectiveDecoderConfig(name="selected_units", attention_size=64, source_encoder_name='marine_encoder'),
        SingleSelectiveDecoderConfig(name="target", attention_size=64, source_encoder_name='enemy_encoder', mask="enemy_mask"),
        CategoricalDecoderConfig(name="pos_x", n=3, hidden_layer_sizes=[512, 256]),
        CategoricalDecoderConfig(name="pos_y", n=3, hidden_layer_sizes=[512, 256]),
    ],
    aggregator=DenseAggregatorConfig(hidden_layer_sizes=[512, 256]),
    value=ValueApproximatorConfig(hidden_layer_sizes=[64, 32]),
)

model_config = {
    "minigame_model": {
        "class": PPOPolicy,  # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network_config": network_cfg,  # 神经网络结构
            "device": "cuda",
        },
        "save": {
            "interval": 20,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
    },
}

feature_list = [encoder_cfg.feature_set for encoder_cfg in network_cfg.encoders] + [enemy_mask_feature_set]

pipeline = {
    "minigame_pipeline": {
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

agents = {"minigame_agent": {"model": "minigame_model", "pipeline": "minigame_pipeline"}}


env = {
    "class": MinigameEnv,
    "params": {},
}

builder = ExBuilder(agents, model_config, env, pipeline)
print(builder)
