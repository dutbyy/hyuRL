from hyurl.network import (
    CommonEncoder,
    DenseAggregator,
    ValueApproximator,
    CategoricalDecoder,
    ComplexNetwork,
)


def get_cartpole_network_config():
    network_cfg = {
        "encoder": {
            "class": CommonEncoder,
            "params": {
                "in_features": 4,
                "hidden_layer_sizes": [128, 128],
                "output_size": 256,
            },
            "inputs": ["observation"],
        },
        "aggregator": {
            "class": DenseAggregator,
            "params": {
                "in_features": 256,
                "hidden_layer_sizes": [256, 128],
                "output_size": 256,
            },
            "inputs": ["encoder"],
        },
        "value": {
            "class": ValueApproximator,
            "params": {
                "in_features": 256,
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ["aggregator"],
        },
        "action": {
            "class": CategoricalDecoder,
            "params": {
                "in_features": 256,
                "n": 2,
                "hidden_layer_sizes": [128, 64],
            },
            "inputs": ["aggregator"],
        },
    }
    return network_cfg


def get_model():
    network_cfg = get_cartpole_network_config()
    model = ComplexNetwork(network_cfg)
    return model
