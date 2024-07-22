import gym
from src.memory.buffer import Memory
from src.tools.common import construct, timer_decorator


class LocalTrainer:
    def __init__(self, delayed_policy):
        self.policy = construct(delayed_policy)
        self.policy.train_mode()
        self.memory_buffer = Memory()
        self.env = gym.make("CartPole-v1", max_episode_steps=500)

    @timer_decorator
    def train(self, train_data):
        self.policy.train(train_data)

    def update_state(self):
        pass

    def get_state_dict(self):
        weights = self.policy._network.state_dict()
        return weights
