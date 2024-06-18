import torch
import torch.nn as nn

class Aggregator(nn.Module):
    """用于聚合多个encoder的输出, 如果有rnn的话，通常放在这个组件
    """

    def forward(self, inputs, initial_state=None):
        """
        Args:
            inputs(list(torch.Tensor)): 长度不限的list
            initial_state(torch.Tensor/None): rnn中的initial state
        Returns:
            outputs(torch.Tensor): 维度为[batch_size, n]
            state(Optional[torch.Tensor]): 可选返回，可以返回一个Tensor，也可以
                返回None。state对应rnn中的state，通常，没有用到rnn的时候，返回None。
        """
        raise NotImplementedError
