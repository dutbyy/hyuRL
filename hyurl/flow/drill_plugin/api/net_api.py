from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from hyurl.feature.feature_set import FeatureSet
from hyurl.network.aggregator import aggregator


@dataclass
class NetConfig:
    name: str = ""
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [256, 128])

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
    feature_set: FeatureSet

    @property
    def length(self):
        return self.feature_set.shape[0]


@dataclass
class SpatialEncoderConfig(EncoderConfig):
    channel_num: int = 32
    transformer: Any = None
    down_samples: List[int] = None
    res_block_num: int = 4

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
    name: str = "dense_aggregator"
    activation: str = "relu"


@dataclass
class CommanderNetworkConfig:
    encoders: List[EncoderConfig]
    decoders: List[DecoderConfig]
    aggregator: AggregatorConfig = DenseAggregatorConfig()
    value: ValueApproximatorConfig = ValueApproximatorConfig()
