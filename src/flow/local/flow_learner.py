from hyuRL.src.algo.PPOPolicy import PPOPolicy
import gym
from hyuRL.src.memory.buffer import Memory
from hyuRL.src.tools.common import construct, timer_decorator
from typing import Dict
from hyuRL.src.tools.common import timer_decorator

# class LocalLearner:
#     def __init__(self, flow_config: Dict, model_name: str, builder):
#         self.flow_model = flow_config['algorithm']['flow_model'](model_name, builder)
#         self.flow_model.setstate_learn([self.flow_model._model_name, builder, self.flow_model._model._network.state_dict()])
#         self.flow_model._model._network.cuda()
#         self.flow_model._model.train_mode()

#     @timer_decorator
#     def train(self, train_data):

#         return self.flow_model.learn(train_data)

#     def get_weights(self):
#         return self.flow_model.get_weights()

class LocalLearner:
    def __init__(self, flow_config: Dict):
        self.builder = flow_config['builder']
        self.model_names = self.builder.model_names
        self.flow_model_dic = {}
        for model_name in self.model_names:
            flow_model = flow_config['algorithm']['flow_model'](model_name, self.builder)
            flow_model.setstate_learn([flow_model._model_name, self.builder, flow_model._model._network.state_dict()])
            flow_model._model._network.cuda()
            flow_model._model.train_mode()
        self.flow_model_dic[model_name] = flow_model


    @timer_decorator
    def train(self, model_name, train_data):
        return self.flow_model_dic[model_name].learn(train_data)

    def get_weights(self, model_name):
        return self.flow_model_dic[model_name].get_weights()
