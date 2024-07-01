import gym
from lib.memory.buffer import Memory
from lib.tools.common import construct


class LocalTrainer:
    def __init__(self, delayed_policy):
        self.policy = construct(delayed_policy)
        self.memory_buffer = Memory()
        self.env = gym.make("CartPole-v1", max_episode_steps=500)

    def train(self, train_data):
        self.policy.train(train_data)

    def update_state(self):
        pass

    def get_state_dict(self):
        weights = self.policy._network.state_dict()
        return weights
