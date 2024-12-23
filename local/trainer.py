from hyuRL.src.algo.PPOPolicy import PPOPolicy
import gym
from src.memory.buffer import Memory
from src.tools.common import construct, timer_decorator


class LocalTrainer:
    def __init__(self, delayed_policy):
        self.policy: PPOPolicy = construct(delayed_policy)
        self.policy.train_mode()

    @timer_decorator
    def train(self, train_data):
        return self.policy.learn(train_data)

    def update_state(self):
        pass

    def get_state_dict(self):
        weights = self.policy._network.state_dict()
        return weights
