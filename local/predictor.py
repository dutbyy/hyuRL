import gym
from lib.memory.buffer import Memory, Fragment
import torch
from multiprocessing import Pool

class LocalPredictor:
    def __init__(self):
        self.memory_buffer = Memory()
        self.env = gym.make("CartPole-v1", max_episode_steps=500)
    
    def predict(self, batch_size):
        pass
    
    def train(self):
        pass