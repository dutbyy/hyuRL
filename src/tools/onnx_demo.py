
# from src.network import *
from src.tools.common import construct
import torch
from src.network.complex import ComplexNetwork
from collections import OrderedDict


def main1():
    network_cfg = {
        "encoder_demo": {
            "class": CommonEncoder,
            "params": {
                "in_features": 16,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_a", "feature_b"],
        },
        "encoder_demo0": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_b"],
        },
        "encoder_demo1": {
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
            "inputs": ["encoder_demo", "encoder_demo1", "encoder_demo0"],
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features": 256,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["aggregator"],
        },
        "action1": {
            "class": CategoricalDecoder,
            "params": {
                "n": 2,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["aggregator"],
        },
        "action2": {
            "class": GaussianDecoder,
            "params": {
                "n": 10,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["action1"],
        },
    }
    delayed_policy = {
        "class": ComplexNetwork,
        "params": {"network_config": network_cfg},
    }

    dynamic_axes = {
        "feature_a": {0: "batch", 1: "feature_a"},
        "feature_b": {0: "batch", 1: "feature_b"},
        "feature_c": {0: "batch", 1: "feature_c"},
    }
    model = construct(delayed_policy)
    dummy_input = OrderedDict()
    dummy_input["feature_a"] = torch.rand(1, 4)
    dummy_input["feature_b"] = torch.rand(1, 12)
    dummy_input["feature_c"] = torch.rand(1, 12)
    outputs = model(dummy_input)
    print("outputs is ", outputs)
    input_names = [i for i in dummy_input.keys()]
    output_names = [
        "value",
        "logits1",
        "logits2_mu",
        "logits2_std",
        "action1",
        "action2",
    ]
    # output_names=output_names,
    print(dummy_input)
    # torch.onnx.export(
    #     model,
    #     (dummy_input, None),
    #     "./model.onnx",
    #     verbose=True,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=dynamic_axes,
    # )
    torch.onnx.dynamo_export(
        model,
        (dummy_input, None)
    ).save("dynamo_model.onnx")

def main2():
    import onnx
    import onnxruntime
    import numpy as np

    ort_session = onnxruntime.InferenceSession("./model.onnx")

    input_name = ort_session.get_inputs()[0].name
    out_names = ort_session.get_outputs()
    in_names = ort_session.get_inputs()
    input_data = np.random.randn(1, 4).astype(np.float32)  # 用实际输入数据替换
    ort_outputs = ort_session.run(None, {input_name: input_data})
    # pytorch_outputs = model(torch.from_numpy(input_data))
    # print([i.name for i in in_names])
    print(input_name)
    print([i.name for i in out_names])
    print(ort_outputs)

def main3(buffer):
    import onnx
    import onnxruntime
    import numpy as np

    print("Checking ONNX model...")
    onnx_model = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(buffer.getvalue())

    input_name = ort_session.get_inputs()[0].name
    out_names = ort_session.get_outputs()
    in_names = ort_session.get_inputs()
    input_data = np.random.randn(1, 4).astype(np.float32)  # 用实际输入数据替换
    ort_outputs = ort_session.run(None, {input_name: input_data})
    # pytorch_outputs = model(torch.from_numpy(input_data))
    # print([i.name for i in in_names])
    print(input_name)
    print([i.name for i in out_names])
    print(ort_outputs)

def main4():
    from torchview import draw_graph

    network_cfg = {
        "encoder_demo": {
            "class": CommonEncoder,
            "params": {
                "in_features": 16,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_a", "feature_b"],
        },
        "encoder_demo0": {
            "class": CommonEncoder,
            "params": {
                "in_features": 12,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ["feature_b"],
        },
        "encoder_demo1": {
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
            "inputs": ["encoder_demo", "encoder_demo1", "encoder_demo0"],
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features": 256,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["aggregator"],
        },
        "action1": {
            "class": CategoricalDecoder,
            "params": {
                "n": 2,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["aggregator"],
        },
        "action2": {
            "class": GaussianDecoder,
            "params": {
                "n": 10,
                "hidden_layer_sizes": [64],
            },
            "inputs": ["action1"],
        },
    }
    delayed_policy = {
        "class": ComplexNetwork,
        "params": {"network_config": network_cfg},
    }
    import onnx

    onnx_model = onnx.load_model("./model.onnx")

    model_graph = draw_graph(
        onnx_model,
        input_data={
            "input_dict": {
                "feature_a": torch.rand(2, 4),
                "feature_b": torch.rand(2, 12),
                "feature_c": torch.rand(2, 12),
            }
        },
        device="cpu",
    )
    model_graph.visual_graph


if __name__ == "__main__":
    fix_print()
    # buffer = main()
    main1()
    # main2()

    # main3(buffer)
    # main4()

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

