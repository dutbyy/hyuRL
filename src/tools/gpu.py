import numpy as np
import torch
from src.memory.buffer import Fragment

def auto_move(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    if isinstance(data, np.floating):
        return data
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = auto_move(v, device)
        return data
    elif isinstance(data, list):
        return [auto_move(item, device) for item in data]
    elif isinstance(data, Fragment):
        data.__dict__ = auto_move(data.__dict__, device)
        return data
    else : 
        return data
        # raise ValueError(f"Automove unsupport Type of value : {type(data)}")
        

        
    
