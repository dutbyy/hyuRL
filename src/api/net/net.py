from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from hyuRL.src.feature.feature_set import FeatureSet, CommonFeatureSet, EntityFeatureSet, SpatialFeatureSet
from hyuRL.src.feature.feature import Feature, VectorFeature, OnehotFeature
from hyuRL.src.network.aggregator import aggregator

@dataclass
class NetConfig:
    name: str = ""
    hidden_layer_sizes: List[int] = field(default_factory = lambda : [])
    dependency: List[str] = None
    def get_name(self):
        return self.name

@dataclass
class EncoderConfig(NetConfig):
    feature_set: FeatureSet = None
    def get_name(self):
        return self.name if self.name else f"encoder_{self.feature_set.name}"

    def __post_init__(self):
        if not self.name:
            self.name = f'encoder_{self.feature_set.name}'

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
    source_encoder_name: str = None
    dependency: str = ""
    mask: str = None

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
        return self.feature_set.shape[:-1]

@dataclass
class CategoricalDecoderConfig(DecoderConfig):
    n: int = -1

@dataclass
class GaussianDecoderConfig(DecoderConfig):
    n: int = 1
    activation: str = "relu"

@dataclass
class SingleSelectiveDecoderConfig(DecoderConfig):
    attention_size: int = 64

@dataclass
class ValueApproximatorConfig(NetConfig):
    name: str = "aggregator"

@dataclass
class DenseAggregatorConfig(AggregatorConfig):
    name: str= "dense_aggregator"
    activation: str = "relu"

@dataclass
class CommanderNetworkConfig:
    encoders:   List[EncoderConfig]
    decoders:   List[DecoderConfig]
    aggregator: AggregatorConfig = DenseAggregatorConfig()
    value:      ValueApproximatorConfig = ValueApproximatorConfig()
