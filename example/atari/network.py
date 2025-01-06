import torch
from torch import nn
from torch.distributions import Categorical

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Xavier初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ActorCriticCnn(nn.Module):
    def __init__(self, features_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        n_flatten = 3136
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.value_net = nn.Linear(features_dim, 1)
        self.action_net = nn.Linear(features_dim, 9)
        # 初始化网络参数
        init_weights(self)


    def forward(self, x, behavior_action_dict=None, training=False, deterministic=False,):
        x = x['common'].permute(0, 3, 1, 2)

        feature = self.linear(self.cnn(x))
        value = self.value_net(feature)

        action_logits = self.action_net(feature)
        dist = Categorical(logits=action_logits)
        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()
        # log_prob = distribution.log_prob(actions)
        return {
            "action": {
                "meta_action": actions,
            },
            "logits": {
                "meta_action": action_logits
            },
            "value": value
        }

    def log_probs(self, logits_dict, action_dict, decoder_mask):
        dist = Categorical(logits=logits_dict['meta_action'])
        logp = dist.log_prob(action_dict['meta_action']).squeeze()
        return {
            "meta_action": logp
        }

    def entropy(self, logits_dict, behavior_mask_dict):
        dist = Categorical(logits=logits_dict['meta_action'])
        entropy_dict = {
            "meta_action": dist.entropy().squeeze()
        }
        return entropy_dict
