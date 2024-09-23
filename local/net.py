from src.network import *


network_cfg = {
    "encoder_demo" : {
        "class": CommonEncoder,
        "params": {
            "in_features" : 4,
            "hidden_layer_sizes": [256, 128],
        },
        "inputs": ['feature_a']
    },
    "aggregator": {
        "class": DenseAggregator,
        "params": {
            "in_features" : 128,
            "hidden_layer_sizes": [256, 128],
            "output_size": 256
        },
        "inputs": ['encoder_demo']
    },
    "value_app": {
        "class": ValueApproximator,
        "params": {
            "in_features" : 256,
            "hidden_layer_sizes": [256, 256, 128],
        },
        "inputs": ['aggregator']
    },
    "action" : {
        "class": CategoricalDecoder,
        "params": {
            "n" : 2,
            "hidden_layer_sizes": [128, 128],
        },
        "inputs": ['aggregator']
    }
}


def get_model():
    from src.network import ComplexNetwork
    model = ComplexNetwork(network_cfg)
    # model.to('cuda')
    return model
