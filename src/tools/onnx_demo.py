import torch
import numpy as np
from torch import nn
from collections import OrderedDict

from src.network import CommonEncoder, EntityEncoder, SpatialEncoder
from src.network import CategoricalDecoder, GaussianDecoder
from src.network import DenseAggregator, ValueApproximator
from src.network import ComplexNetwork
from src.tools.common import construct


def fix_print():
    # return
    import builtins
    import os

    origin_print = builtins.print

    def custom_print(*args, **kwargs):
        import datetime
        import inspect

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        prefix = f"[{timestamp}] [{os.path.basename(caller.filename)}:{caller.lineno}]"
        origin_print(prefix, *args, **kwargs)

    builtins.print = custom_print


def nested_from_numpy(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        return {k: nested_from_numpy(v) for k, v in x.items()}


network_cfg = {
    "info_encoder": {
        "class": CommonEncoder,
        "params": {
            "in_features": 4,
            "hidden_layer_sizes": [128, 128],
        },
        "inputs": ["feature_a"],
    },
    "enemy_encoder": {
        "class": EntityEncoder,
        "params": {
            "length": 10,
            "in_features": 12,
            "hidden_layer_sizes": [128, 128],
        },
        "inputs": ["feature_b"],
    },
    "height_map_encoder": {
        "class": SpatialEncoder,
        "params": {
            "in_shape": [16, 16],
            "in_features": 12,
            "channel_num": 32,
            "output_size": 128,
        },
        "inputs": ["feature_c"],
    },
    "aggregator": {
        "class": DenseAggregator,
        "params": {
            "in_features": 128 * 3,
            "hidden_layer_sizes": [256],
            "output_size": 256,
        },
        "inputs": ["info_encoder", "enemy_encoder", "height_map_encoder"],
    },
    "value_app": {
        "class": ValueApproximator,
        "params": {
            "in_features": 256,
            "hidden_layer_sizes": [64],
        },
        "inputs": ["aggregator"],
    },
    "move": {
        "class": CategoricalDecoder,
        "params": {
            "n": 2,
            "hidden_layer_sizes": [64],
        },
        "inputs": ["aggregator"],
    },
    "attack": {
        "class": GaussianDecoder,
        "params": {
            "n": 1,
            "hidden_layer_sizes": [64],
        },
        "inputs": ["move"],
    },
}
delayed_policy = {
    "class": ComplexNetwork,
    "params": {"network_config": network_cfg},
}
dummy_input = OrderedDict(
    {
        "feature_a": np.ones((10, 4)).astype(np.float32),
        "feature_b": np.ones((10, 10, 12)).astype(np.float32),
        "feature_c": np.ones((10, 16, 16, 12)).astype(np.float32),
    }
)


def export_onnx():
    dynamic_axes = {
        "feature_a": {0: "batch"},
        "feature_b": {0: "batch"},
        "feature_c": {0: "batch"},
        "value": {0: "batch"},
        "logits_move": {0: "batch"},
        "logits_attack_mu": {0: "batch"},
        "logits_attack_std": {0: "batch"},
        "action_move": {0: "batch"},
        "action_attack": {0: "batch"},
        "aggregator": {0: "batch"},
    }
    model = construct(delayed_policy)
    global dummy_input
    dm_input = nested_from_numpy(dummy_input)
    outputs = model(dm_input)
    print("outputs is ", outputs)
    input_names = [i for i in dm_input.keys()]
    output_names = [
        "value",
        "logits_move",
        "logits_attack_mu",
        "logits_attack_std",
        "action_move",
        "action_attack",
    ]
    print(dummy_input)
    torch.onnx.export(
        model,
        (dm_input, None),
        "./model.onnx",
        # verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    # torch.onnx.dynamo_export(
    #     model,
    #     (dummy_input, None)
    # ).save("dynamo_model.onnx")


def onnx_info():
    import onnxruntime

    ort_session = onnxruntime.InferenceSession("./model.onnx")
    input_names = [it.name for it in ort_session.get_inputs()]
    output_names = [it.name for it in ort_session.get_outputs()]
    print(f"model input names : {input_names}")
    print(f"model input names : {output_names}")

    input_shape = [f"{it.name}: {it.shape}" for it in ort_session.get_inputs()]
    output_shape = [f"{it.name}: {it.shape}" for it in ort_session.get_outputs()]
    print(f"model input shapes : {input_shape}")
    print(f"model input shapes : {output_shape}")


def onnx_check():
    import onnx

    buffer = None
    onnx_model = onnx.load_model("./model.onnx")
    onnx.checker.check_model(onnx_model)


def onnx_run():
    import onnxruntime
    import numpy as np

    ort_session = onnxruntime.InferenceSession("./model.onnx")

    dummy_input = {}
    dummy_input["feature_a"] = np.random.normal(size=(1, 4)).astype(np.float32)
    dummy_input["feature_b"] = np.random.normal(size=(1, 12)).astype(np.float32)
    dummy_input["feature_c"] = np.random.normal(size=(1, 12)).astype(np.float32)
    # print(dummy_input)
    ort_outputs = ort_session.run(None, dummy_input)
    output_names = [it.name for it in ort_session.get_outputs()]
    assert len(ort_outputs) == len(output_names)
    outputs = {k: v for k, v in zip(output_names, ort_outputs)}
    print(outputs)


def onnx_view():
    from torchview import draw_graph

    model: nn.Module = construct(delayed_policy).eval()
    model.requires_grad_(False)
    dm_input = nested_from_numpy(dummy_input)
    model(dm_input)
    model_graph = draw_graph(
        model,
        input_data={"input_dict": dm_input},
        device="cpu",
        save_graph=True,
        expand_nested=True,
    )
    # model_graph.visual_graph


if __name__ == "__main__":
    # fix_print()
    export_onnx()
    # onnx_info()
    # onnx_check()
    # onnx_run()
    onnx_view()
