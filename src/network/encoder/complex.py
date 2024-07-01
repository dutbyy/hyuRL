import torch 
from torch import nn
from typing import TYPE_CHECKING, Any, Dict, List

def is_class_dict(class_dict: Dict):
    return ("class" in class_dict) and ("params" in class_dict)

def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件
    """

    if not is_class_dict(class_dict):
        raise ValueError(f"Expected a dict with keys 'class' and 'params', but got {class_dict}")

    class_ = class_dict["class"]
    params = class_dict["params"]
    return class_(**params)

class ComplexEncoder(nn.Module):
    def __init__(self, encoder_config):    
        super().__init__()
        self._encoder_config = encoder_config    
        self._encoders = nn.ModuleDict({name: construct(encoder) for name, encoder in self._encoder_config.items()})

    
    def forward(self, input_dict: Dict, training:bool = False):
        encoders_output_dict = {} 
        encoder_embedding_dict = {}
        for name, encoder in self._encoders.items():
            outputs, embeddings = encoder(torch.Tensor(input_dict[name]), training)   
            encoders_output_dict[name] = outputs
            encoder_embedding_dict[name] = embeddings
        return encoders_output_dict, encoder_embedding_dict