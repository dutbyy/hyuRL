from hyuRL.src.network import Encoder, CommonEncoder, EntityEncoder, SpatialEncoder, ValueApproximator
from hyuRL.src.network import CategoricalDecoder, GaussianDecoder, SingleSelectiveDecoder
from hyuRL.src.network import Aggregator
from hyuRL.src.network import DenseAggregator

from hyuRL.src.api.net import CommonEncoderConfig, EntityEncoderConfig, SpatialEncoderConfig
from hyuRL.src.api.net import CategoricalDecoderConfig, GaussianDecoderConfig, SingleSelectiveDecoderConfig
from hyuRL.src.api.net import AggregatorConfig
from hyuRL.src.api.net import DenseAggregatorConfig

from ..api.net_api import CommanderNetworkConfig, NetConfig


ConfigMapping = {
    CommonEncoderConfig:            CommonEncoder,
    EntityEncoderConfig:            EntityEncoder,
    SpatialEncoderConfig:           SpatialEncoderConfig,
    CategoricalDecoderConfig:       CategoricalDecoder,
    GaussianDecoderConfig:          GaussianDecoder,
    SingleSelectiveDecoderConfig:   SingleSelectiveDecoder,
    AggregatorConfig:               Aggregator,
    DenseAggregatorConfig:          DenseAggregator,
}

def consturct(net_cfg: NetConfig):
    net_class = ConfigMapping.get(type(net_cfg))
    if isinstance(CommonEncoderConfig):
        pass
    elif isinstance(EntityEncoderConfig):
        pass
    elif isinstance(SpatialEncoderConfig):
        pass
    elif isinstance(CategoricalDecoderConfig):
        pass
    elif isinstance(GaussianDecoderConfig):
        pass
    elif isinstance(SingleSelectiveDecoderConfig):
        pass
    elif isinstance(AggregatorConfig):
        pass
    elif isinstance(DenseAggregatorConfig):
        pass
    else:
        raise Exception("UnsupportNetworks")
    return net_cfg.get_name 

def generate(config = CommanderNetworkConfig):
    DefaultFeatureLength = 256
    module_dict = {}
    for encoder_cfg in config.encoders:
        name = encoder_cfg.get_name()
            
        if isinstance(encoder_cfg, CommonEncoderConfig):
            encoder = CommonEncoder(
                in_features=encoder_cfg.feature_size, 
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes) 
        elif isinstance(encoder_cfg, EntityEncoderConfig):
            encoder = EntityEncoder(
                length=encoder_cfg.length, 
                in_features=encoder_cfg.feature_size, 
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes,
                transformer=encoder_cfg.transformer,
                pooling=encoder_cfg.pooling,
            ) 
        elif isinstance(encoder_cfg, SpatialEncoderConfig):
            encoder = SpatialEncoder(
                in_shape=encoder_cfg.shape, 
                in_features= encoder_cfg.feature_size, 
                channel_num= encoder_cfg.channel_num, 
                output_size= DefaultFeatureLength,
                down_samples= encoder_cfg.down_samples,
                res_block_num= encoder_cfg.res_block_num,
            )
        else:
            raise Exception(f"Not Supported Encoder : {type(encoder_cfg)}")
        module_dict[name] = encoder
    
    
    aggregator = DenseAggregator(
        in_features = DefaultFeatureLength * len(config.encoders),
        hidden_layer_sizes = config.aggregator.hidden_layer_sizes,
        output_size = DefaultFeatureLength
    )
    name = config.aggregator.get_name()
    module_dict[name] = aggregator
    
    value = ValueApproximator(
        in_features = DefaultFeatureLength * len(config.encoders),
        hidden_layer_sizes = config.value.hidden_layer_sizes,
    )
    name = config.value.get_name()
    module_dict[name] = value
    
    
    for decoder_cfg in config.decoders:
        name = decoder_cfg.get_name()
            
        if isinstance(decoder_cfg, CategoricalDecoderConfig):
            decoder = CategoricalDecoder(
                n = decoder_cfg.n,
                in_features = DefaultFeatureLength,
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes) 
        elif isinstance(decoder_cfg, GaussianDecoderConfig):
            decoder = GaussianDecoder(
                n = decoder_cfg.n,
                hidden_layer_sizes=decoder_cfg.hidden_layer_sizes,
            ) 
        elif isinstance(decoder_cfg, SingleSelectiveDecoderConfig):
            decoder = SingleSelectiveDecoder(
                in_features = DefaultFeatureLength,
                attention_size = decoder_cfg.attention_size,
            )
        else:
            raise Exception("Not Supported Encoder")
        module_dict[name] = decoder
    
    for k, v in module_dict.items():
        print(k, v)
