import torch

def Mask(inputs, inputs_len, mode='mul', seq_axis=1):
    """
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, inputs_len, input_size)的张量；
    inputs_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    """
    assert inputs_len is not None, "inputs_len should not be None"
    inputs_shape = inputs.size()
    max_len = inputs_shape[seq_axis]

    # 创建 mask 张量
    mask = torch.arange(max_len, device=inputs.device).unsqueeze(0) < inputs_len.unsqueeze(1)
    mask = mask.to(dtype=inputs.dtype)  # 转换为与 inputs 相同的数据类型

    # 调整 mask 的形状以匹配 inputs
    mask_shape = mask.size()
    mask_new_shape = [1] * len(inputs_shape)
    mask_new_shape[0] = mask_shape[0]  # batch_size
    mask_new_shape[seq_axis] = mask_shape[1]  # seq_len
    mask = mask.view(mask_new_shape)

    # 根据 mode 处理 inputs
    if mode == 'mul':
        outputs = inputs * mask
    elif mode == 'add':
        outputs = inputs - (1 - mask) * 1e12
    else:
        raise ValueError(f"mode {mode} is not supported")

    return outputs