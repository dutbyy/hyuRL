EPOCH = 2
import logging
import timeit
logging.getLogger().setLevel(10)
import numpy as np
from sympy import loggamma
import torch
from torch import optim
import gym
from network.template import TemplateNetwork
from loss.vpg import VPGLoss

from tools.gae import calculate_gae
from memory.buffer import Memory
is_cuda = torch.cuda.is_available()
is_cuda = False
device = torch.device("cuda" if is_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)
from tools.gpu import auto_move

class AgentPPO:
    def __init__(self, *args):
        self._network = TemplateNetwork(*args)
        self._network.to(device)
        self._optimizer = optim.Adam(self._network.parameters(), lr=1e-4)
        self._optimizer_value = optim.Adam(self._network._value.parameters(), lr=1e-4)
        self._loss_fn = VPGLoss(entropy_coef=0.0)
        self.memory = Memory()
        
    def inference(self, state):
        inputs = {"common_encoder": torch.Tensor(state)}
        inputs = auto_move(inputs, device)
        with torch.no_grad():
            outputs = self._network(inputs)
        return outputs

    def train(self):
        trainning_data  =  self.memory.get_batch(1024)
        trainning_data = auto_move(trainning_data, device)
        inputs_dict = trainning_data['states']
        behavior_action_dict = trainning_data.get("actions")
        # behavior_logits_dict = trainning_data.get("logits")
        behavior_mask_dict = trainning_data.get("masks")
        behavior_values = trainning_data.get("values")
        advantages = trainning_data.get("advantages")
        old_logp_dict_running = trainning_data.get('log_probs')
        old_logp = sum(old_logp_dict_running.values())  # 在动作维度合并logp (相当于动作概率连乘)
        for epoch in range(6):
            predict_output_dict = self._network(inputs_dict, behavior_action_dict, training=True)
            logits_dict = predict_output_dict['logits']
            
            logp_dict = self._network.log_probs(logits_dict, behavior_action_dict, behavior_mask_dict)
            logp = sum(logp_dict.values())

            # 计算当前策略的熵
            entropy_dict = self._network.entropy(logits_dict, behavior_mask_dict)
            entropy = torch.mean(sum(entropy_dict.values()))
            value = predict_output_dict['value']
            target_value = advantages + behavior_values
            loss, policy_loss, value_loss = self._loss_fn(
                    old_log_prob=old_logp,
                    log_prob=logp,
                    advantage=advantages,
                    old_value=behavior_values,
                    value=value,
                    target_value=target_value,
                    entropy=entropy)

            self._optimizer.zero_grad()
            if epoch == 0:
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
                self._optimizer.step()
            else : 
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network._value.parameters(), 40.0)
                self._optimizer_value.step()
        return 
            
            
if __name__ == '__main__':
    from network.encoder.common import CommonEncoder
    from network.decoder.categorical import CategoricalDecoder
    from network.value import ValueApproximator
    from network.aggregator.dense import DenseAggregator
    env = gym.make("CartPole-v1", max_episode_steps=500)

    
    encoder_config  = {
        "common_encoder" : {
            "class": CommonEncoder,
            "params": {
                "in_features" : 4, 
                "hidden_layer_sizes": [256, 128],
            }
        } 
    }
    decoder_config = {
        "action" : {
            "class": CategoricalDecoder,
            "params": {
                "n" : 2,
                "hidden_layer_sizes": [256, 128],
                # "temperature": 0.5,
            }
        } 
    }
    aggregator_config = {
        "class": DenseAggregator,
        "params": {
            "in_features" : 128, 
            "hidden_layer_sizes": [256, 128],
            "output_size": 256,
        }
    } 
    value_config = {
       "class": ValueApproximator,
       "params": {
           "hidden_layer_sizes": [256],
       }    
    }
    
    agent = AgentPPO(encoder_config, aggregator_config, decoder_config, value_config)
    EpisodeNum = 5000
    
    avg_reward = 0
    train_step = 0
    total_rewards = []
    import timeit
    for episode in range(EpisodeNum):
        inputs_dict = {}
        state, _ = env.reset()
        total_reward = 0
        # print("Reset The Environment")
        while True:
            outputs = agent.inference(state)
            nstate, reward, done, truncted, info = env.step(outputs['action']['action'].item())
            # reward -= abs(nstate[0]) * .1
            # reward -= abs(nstate[1]) * .1
            # reward -= abs(nstate[2]) * .3
            # reward -= abs(nstate[3]) * .3
            reward *= .01
            total_reward += 1
            done = done or truncted

            
            action_mask = {"action": torch.tensor(1)}
            logp = agent._network.log_probs( outputs['logits'], outputs['action'], action_mask)
            agent.memory.store( {"common_encoder": torch.Tensor(state)}, outputs['action'], reward, 1 if done else 0 , logp, action_mask, outputs['value'], outputs['logits'])
           
            if agent.memory.size >= 1024:
                print(f'{train_step}: average reward is {sum(total_rewards)/len(total_rewards)}', end = '\t') 
                start_time = timeit.default_timer()
                agent.train()
                end_time = timeit.default_timer()

                print(f"train eplased time : {end_time - start_time}")
                train_step +=1
                agent.memory.reset()
                total_rewards = []
                break
            if done :
                total_rewards.append(total_reward)
                break   
            state = nstate

            
            # agent.clean_data()
