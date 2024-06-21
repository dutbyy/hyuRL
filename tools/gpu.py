import torch
def auto_move(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = auto_move(v, device)
        return data
    elif isinstance(data, list):
        return [auto_move(item, device) for item in data]
    else : 
        raise ValueError(f"Automove unsupport Type of value : {type(data)}")
        
        
    
