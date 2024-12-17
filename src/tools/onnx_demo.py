from multiprocessing import dummy
import torch
import numpy as np
from src.network import *
from src.tools.common import construct
from src.network.complex import ComplexNetwork
from collections import OrderedDict

def fix_print():
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

nested_from_numpy = lambda x : torch.from_numpy(x) if isinstance(x, np.ndarray) else x if isinstance(x, torch.Tensor) else { k:nested_from_numpy(v)  for k, v in x.items()}

def export_onnx():
    network_cfg = {
        "encoder_demo_a": {
            "class": CommonEncoder,
            "params": {
                "in_features": 16,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_a", "feature_b"],
        },
        "encoder_demo_b": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_b"],
        },
        "encoder_demo_c": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
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
            "inputs": ["encoder_demo_a", "encoder_demo_b", "encoder_demo_c"],
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
                "n": 3,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["move"],
        },
    }
    delayed_policy = {
        "class": ComplexNetwork,
        "params": {"network_config": network_cfg},
    }

    dynamic_axes = {
        "feature_a":    {0: "batch"},
        "feature_b":    {0: "batch"},
        "feature_c":    {0: "batch"},
        "value":        {0: "batch"},
        "logits_move":  {0: "batch"},
        "logits_attack_mu":     {0: "batch"},
        "logits_attack_std":    {0: "batch"},
        "action_move":          {0: "batch"},
        "action_attack":        {0: "batch"},
        "aggregator":           {0: "batch"},
    }
    model = construct(delayed_policy)
    dummy_input = OrderedDict()
    # dummy_input["feature_a"] = np.random.normal(size=(10, 4)).astype(np.float32)
    # dummy_input["feature_b"] = np.random.normal(size=(10, 12)).astype(np.float32)
    # dummy_input["feature_c"] = np.random.normal(size=(10, 12)).astype(np.float32)
    dummy_input["feature_a"] = np.ones((10, 4)).astype(np.float32)
    dummy_input["feature_b"] = np.ones((10, 12)).astype(np.float32)
    dummy_input["feature_c"] = np.ones((10, 12)).astype(np.float32)
    # nested_from_numpy = lambda x : torch.from_numpy(x) if isinstance(x, np.ndarray) else x if isinstance(x, torch.Tensor) else OrderedDict({ k:nested_from_numpy(v)  for k, v in x.items()})
    dummy_input = nested_from_numpy(dummy_input)
    # dummy_input["feature_a"] = torch.rand(size=(1, 4))
    # dummy_input["feature_b"] = torch.rand(size=(1, 12))
    # dummy_input["feature_c"] = torch.rand(size=(1, 12))
    outputs = model(dummy_input)
    print("outputs is ", outputs)
    input_names = [i for i in dummy_input.keys()]
    output_names = [
        "value",
        "logits_move",
        "logits_attack_mu",
        "logits_attack_std",
        "action_move",
        "action_attack",
        "aggregator",
    ]
    # output_names=output_names,
    print(dummy_input)
    torch.onnx.export(
        model,
        (dummy_input, None),
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
    input_names = [ it.name for it in  ort_session.get_inputs()]
    output_names = [ it.name for it in  ort_session.get_outputs()]
    print(f"model input names : {input_names}")
    print(f"model input names : {output_names}")
    
    input_shape = [ f"{it.name}: {it.shape}" for it in  ort_session.get_inputs()]
    output_shape = [ f"{it.name}: {it.shape}" for it in  ort_session.get_outputs()]
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
    output_names = [ it.name for it in  ort_session.get_outputs()]
    assert len(ort_outputs) == len(output_names)
    outputs = {
        k: v
        for k, v in zip(output_names, ort_outputs)
    }
    print(outputs)
    
    
def onnx_view():
    from torchview import draw_graph
    network_cfg = {
        "encoder_demo_a": {
            "class": CommonEncoder,
            "params": {
                "in_features": 4,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_a"],
        },
        "encoder_demo_b": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_b"],
        },
        "encoder_demo_c": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
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
            "inputs": ["encoder_demo_a", "encoder_demo_b", "encoder_demo_c"],
            # "inputs": ["encoder_demo_c"],
        },
        # "value_app": {
        #     "class": ValueApproximator,
        #     "params": {
        #         "in_features": 256,
        #         "hidden_layer_sizes": [64],
        #     },
        #     "inputs": ["aggregator"],
        # },
        # "move": {
        #     "class": CategoricalDecoder,
        #     "params": {
        #         "n": 2,
        #         "hidden_layer_sizes": [64],
        #     },
        #     "inputs": ["aggregator"],
        # },
        # "attack": {
        #     "class": GaussianDecoder,
        #     "params": {
        #         "n": 3,
        #         "hidden_layer_sizes": [64],
        #     },
        #     "inputs": ["move"],
        # },
    }
    delayed_policy = {
        "class": ComplexNetwork,
        "params": {"network_config": network_cfg},
    }
    model = construct(delayed_policy)
    dummy_input = {
        "feature_a": torch.rand(2, 4),
        "feature_b": torch.rand(2, 12),
        "feature_c": torch.rand(2, 12),
    }
    model_graph = draw_graph(
        model,
        input_data={
            "input_dict": dummy_input
        },
        device="cpu",
        save_graph=True,
        expand_nested=True
    )
    # model_graph.visual_graph


if __name__ == "__main__":
    fix_print()
    # export_onnx()
    # onnx_info()
    # onnx_check()
    # onnx_run()
    onnx_view()