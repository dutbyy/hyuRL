
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

    from torchview import draw_graph
    model.requires_grad_(False)
    from collections import OrderedDict
    
    def nested_from_numpy(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            return {k: nested_from_numpy(v) for k, v in x.items()}

    dummy_input = OrderedDict(
        {
            "common": np.ones((64, 10)).astype(np.float32),
            "enemies": np.ones((64, 10, 16)).astype(np.float32),
            "heights": np.ones((64, 16, 16, 10)).astype(np.float32),
        }
    )
    dm_i = nested_from_numpy(dummy_input)
    model(dm_i)
    model_graph = draw_graph(
        model,
        input_data={"input_dict": dm_i},
        device="cpu",
        save_graph=True,
        directory='./tmp',
        filename='torch_view',
        expand_nested=True,
    )
    
    