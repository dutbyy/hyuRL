import torch
from torch import nn
import inspect


def make_active_layer(activation):
    ActiveLayerMap = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if isinstance(activation, str):
        cls = ActiveLayerMap.get(activation, nn.ReLU)
        return cls()
    elif inspect.isclass(activation) and issubclass(activation, nn.Module):
        return activation()
    else:
        raise Exception("Unsupported active function, please use [relu] or [tanh] or [torch's active layer class]")
