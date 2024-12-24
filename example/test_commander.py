
from hyuRL.src.feature.feature import *
from hyuRL.src.feature.feature_set import *
from hyuRL.src.api.net.net import *
from hyuRL.src.network.commander import ComplexNetwork
import torch

if __name__ == "__main__":

    network_cfg = CommanderNetworkConfig(
        encoders = [
            CommonEncoderConfig(
                feature_set = CommonFeatureSet(
                    name = "common",
                    feature_dict = {
                        "space": VectorFeature(10)
                    }
                )
            ),
            EntityEncoderConfig(
                feature_set = EntityFeatureSet(
                    name = "enemies",
                    max_length = 10,
                    feature_dict = {
                        "space": VectorFeature(16)
                    }
                )
            ),
            SpatialEncoderConfig(
                feature_set = SpatialFeatureSet(
                    name = "heights",
                    shape = [16, 16],
                    feature_dict = {
                        "space": VectorFeature(10)
                    }
                )
            ),
        ],
        decoders = [
            CategoricalDecoderConfig(name="meta_action", n=3),
            GaussianDecoderConfig(name="direction", n=1),
            SingleSelectiveDecoderConfig(name="target", source_encoder_name="encoder_enemies"),
        ]
    ) 
    model = ComplexNetwork(network_cfg)

    state = model.state_dict()
    # print(type(state))
    # for k,v in state.items():
    #     print(k, type(v))
    
    for item in model.parameters():
        print(type(item))
        
    print(len([it.detach().numpy() for it in model.parameters()]))
    print(len(model.state_dict()))
    