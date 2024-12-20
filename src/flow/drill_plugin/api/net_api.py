from __future__ import annotations

from flatbuffers.flexbuffers import Object
# from pydantic import BaseModel
BaseModel = Object
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
# from feature_set import FeatureSet
from src.feature.feature_set import FeatureSet, CommonFeatureSet, EntityFeatureSet, SpatialFeatureSet
from src.feature.feature import Feature, VectorFeature
from src.network.aggregator import aggregator

@dataclass
class NetConfig(BaseModel):
    name: str = ""
    hidden_layer_sizes: List[int] = field(default_factory = lambda : [256,  128])

    def get_name(self):
        return self.name

@dataclass
class EncoderConfig(NetConfig):
    feature_set: FeatureSet = None
    def get_name(self):
        return self.name if self.name else f"encoder_{self.feature_set.name}"
    
    @property
    def feature_size(self):
        return self.feature_set.length
    
    def __post_init__(self):
        if self.feature_set is None:
            raise ValueError("feature_set must be provided")
    
@dataclass
class AggregatorConfig(NetConfig):
    name: "aggregator"

@dataclass
class DecoderConfig(NetConfig):
    dependency: str = ""

@dataclass
class CommonEncoderConfig(EncoderConfig, NetConfig):
    # feature_set: FeatureSet 
    pass

@dataclass
class EntityEncoderConfig(EncoderConfig):
    transformer: Any = None
    pooling: Any = None
    feature_set : FeatureSet
    
    @property
    def length(self):
        return self.feature_set.shape[0]
    
@dataclass
class SpatialEncoderConfig(EncoderConfig):
    channel_num: int = 32
    transformer: Any = None
    down_samples: List[int] = None
    res_block_num : int = 4
    
    @property
    def shape(self):
        return self.feature_set.shape
    
@dataclass
class CategoricalDecoderConfig(DecoderConfig):
    n: int = -1

@dataclass
class GaussianDecoderConfig(DecoderConfig):
    n: int = 1
    activation: str = "relu"

@dataclass
class SingleSelectiveDecoderConfig(DecoderConfig):
    source_encoder_embedding: str = None
    attention_size: int = 64

@dataclass
class ValueApproximatorConfig(NetConfig):
    name: str = "aggregator"

@dataclass
class DenseAggregatorConfig(AggregatorConfig):
    name: str= "dense_aggregator"
    activation: str = "relu"

@dataclass
class CommanderNetworkConfig(BaseModel): 
    encoders:   List[EncoderConfig] 
    decoders:   List[DecoderConfig]
    aggregator: AggregatorConfig = DenseAggregatorConfig()
    value:      ValueApproximatorConfig = ValueApproximatorConfig()


# if __name__ == '__main__':
#     from src.flow.drill_plugin.api.net_api import *
#     netconfig = CommanderNetworkConfig(
#         encoders = [
#             CommonEncoderConfig(
#                 feature_set = CommonFeatureSet(
#                     name = "common",
#                     feature_dict = {
#                         "space": VectorFeature(10)
#                     }
#                 )
#             ),
#             EntityEncoderConfig(
#                 feature_set = EntityFeatureSet(
#                     name = "enemies",
#                     max_length = 10,
#                     feature_dict = {
#                         "space": VectorFeature(10)
#                     }
#                 )
#             ),
#             SpatialEncoderConfig(
#                 feature_set = SpatialFeatureSet(
#                     name = "heights",
#                     shape = [128, 128],
#                     feature_dict = {
#                         "space": VectorFeature(10)
#                     }
#                 )
#             ),
#         ],
#         decoders = [
#             CategoricalDecoderConfig(name="meta_action", n=3),
#             GaussianDecoderConfig(name="direction", n=1),
#             SingleSelectiveDecoderConfig(name="target", source_encoder_embedding="enemies"),
#         ]
#     )
#     from src.flow.drill_plugin.config.demo import generate
#     ret = generate(netconfig)