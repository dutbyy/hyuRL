import torch 

is_cuda = torch.cuda.is_available()
# is_cuda = False
device = torch.device("cuda" if is_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

def auto_move(data):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = auto_move(v)
        return data
    elif isinstance(data, list):
        return [auto_move(item) for item in data]
    else : 
        raise ValueError(f"Automove unsupport Type of value : {type(data)}")
        
        
    
