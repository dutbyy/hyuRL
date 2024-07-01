import torch
import torch.nn as nn
from torch.distributions import Distribution

class Decoder(nn.Module):
    """
    The Decoder typically serves as the exit of the entire network and there can be multiple decoders. 
    The main function of the Decoder is to make decisions and output actions. 
    When there are multiple decoders, one decoder's decision can be conditioned on the decisions of other decoders.
    """

    def __init__(self, in_features=None, hidden_state=None, out_features=None):
        self.source_encoder_name = None
        # Initialize your layers here
        super().__init__()

    def forward(self, inputs, action_mask=None, behavior_action=None):
        # Implement your forward pass here
        raise NotImplementedError

    def distribution(self, logits):
        # Define the distribution used for sampling here
        raise NotImplementedError
