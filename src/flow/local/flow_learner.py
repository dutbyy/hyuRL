from hyuRL.src.algo.PPOPolicy import PPOPolicy
import gym
from src.memory.buffer import Memory
from src.tools.common import construct, timer_decorator
from typing import Dict

class LocalLearner:
    def __init__(self, flow_config: Dict, model_name: str, builder):
        self.flow_model = flow_config['algorithm']['flow_model'](model_name, builder)
        # self.flow_model.setstate_learn(self.flow_model._model_name, builder, self.flow_model._model._network.state_dict())
        self.flow_model.setstate_learn([self.flow_model._model_name, builder, self.flow_model._model._network.state_dict()])
        self.flow_model._model._network.cuda()
    def train(self, train_data):
        # print(train_data.keys())
        return self.flow_model.learn(train_data)

    def get_weights(self):
        return self.flow_model.get_weights()