
from network.template import TemplateNetwork
from loss.ppo import PPOLoss
import torch
from torch import nn
from torch import optim
import gym
from tools.gae import calculate_gae
import logging

is_cuda = torch.cuda.is_available()
# is_cuda = False
device = torch.device("cuda" if is_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

class PPOTrainer:
    def __init__(self, *args):
        self._network = TemplateNetwork(*args)
        print("!!!!", next(self._network.parameters()).is_cuda)
        self._network.to(device)
        print('~~~~', next(self._network.parameters()).is_cuda)
        self._optimizer = optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss()
        self.clean_data()
        
    def clean_data(self):
        self.input_dict = {
            "common_encoder": []
        }
        self.behavior_info_dict = {
            "action": [],
            "logits": [],
            "advantage": [],
            "value": [],
            "done": [],
            "decoder_mask": []
        }
        
    def inference(self, state):
        inputs = {"common_encoder": torch.Tensor(state)}
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._network(inputs)
        action = outputs['action']
        # print(f"action is {action}")
        # self.input_dict['common_encoder'].append(torch.Tensor(state))

        return outputs

    
    def train(self):# , inputs_dict, behavior_info_dict):
        # 从behavior_info_dict中提取信息
        inputs_dict = {k : torch.stack(v) for k, v in self.input_dict.items() }
        inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
        behavior_info_dict = self.behavior_info_dict 
        behavior_action = behavior_info_dict['action']
        behavior_action_dict = {}
        for taction in behavior_action:
            for k, v in taction.items():
                if k not in behavior_action_dict:
                    behavior_action_dict[k] = []
                behavior_action_dict[k].append(v)
        behavior_action_dict = {k : torch.stack(v) for k, v in behavior_action_dict.items() }
        # print("behaviour action is , ", behavior_action_dict)
        
        behavior_logits_dict = {}
        for tlogits in behavior_info_dict['logits']:
            for k, v in tlogits.items():
                if k not in behavior_logits_dict:
                    behavior_logits_dict[k] = []
                behavior_logits_dict[k].append(v)
        behavior_logits_dict = {k : torch.stack(v) for k, v in behavior_logits_dict.items() }
        
        
        advantage = behavior_info_dict['advantage']
        dones = behavior_info_dict['done']
        behavior_decoder_mask = {}
        for tmask in behavior_info_dict.get("decoder_mask"):
            for k, v in tmask.items():
                if k not in behavior_decoder_mask:
                    behavior_decoder_mask[k] = []
                behavior_decoder_mask[k].append(v)
        
        behavior_decoder_mask = {k : torch.stack(v).to(device) for k, v in behavior_decoder_mask.items() }
        predict_output_dict = self._network(inputs_dict, behavior_action_dict, training=True)
        logits_dict = predict_output_dict['logits']
        

        # behavior_decoder_mask.to(device)
        # 计算 behavior action 在 new policy 下的 -logp
        logp_dict = self._network.log_probs(logits_dict, behavior_action_dict, behavior_decoder_mask)
        logp = sum(logp_dict.values())
        
        old_logp_dict = self._network.log_probs(behavior_logits_dict, behavior_action_dict, behavior_decoder_mask)
        old_logp = sum(old_logp_dict.values())
        
        # logging.warning(f"log p is {logp}\nold logp is {old_logp}")
        logging.warning(f"log p diff is {(logp - old_logp).mean()}")
        # 计算熵
        entropy_dict = self._network.entropy(logits_dict, behavior_decoder_mask)
        entropy = torch.mean(sum(entropy_dict.values()))

        
        
        value = predict_output_dict['value'].to(device)
        behavior_value = torch.Tensor(behavior_info_dict['value']).to(device)
        advantage = torch.Tensor(calculate_gae(advantage, behavior_value, dones)).to(device)
        # 计算 target_value 参考：“A Closer Look At Deep Policy Gradients”
        # https://arxiv.org/pdf/1811.02553.pdf
        target_value = advantage + behavior_value[:-1]

        # logging.warning(f"value:        {value}             {value.shape}")
        # logging.warning(f"target_value: {target_value}      {target_value.shape}")
        # logging.warning(f"old_value:    {behavior_value}    {behavior_value.shape}")
        # logging.warning(f"advantage:    {advantage}         {advantage.shape}")
        # normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
              
        loss, policy_loss, value_loss, entropy_loss = self._loss_fn(
                old_log_prob=old_logp,
                log_prob=logp,
                advantage=advantage,
                old_value=behavior_value[:-1],
                value=value,
                target_value=target_value,
                entropy=entropy)
        mean_advantage = torch.mean(advantage)
        # kl_dict = self._network.kl(logits_dict, behavior_logits_dict, behavior_decoder_mask)
        # kl = torch.mean(sum(kl_dict.values()))
        self._optimizer.zero_grad()

        # 反向传播和优化器更新
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 40.0)
        self._optimizer.step()

        summary = {
            "loss": loss.item(),
            # "policy_loss": policy_loss.item(),
            # "value_loss": value_loss.item(),
            # "entropy": entropy.item(),
            # "advantage": advantage.mean().item(),
            # KL散度在PyTorch中的计算可能需要自定义实现，这里省略了
            # "kl": kl,
        }
        # print(f"policy loss : {policy_loss}, value_loss : {value_loss}, entropy: {entropy}, loss: {loss}")
        # self.clean_data()
        return summary
            
            
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
           "hidden_layer_sizes": [256, 128],
       }    
    }
    
    agent = PPOTrainer(encoder_config, aggregator_config, decoder_config, value_config)
    EpisodeNum = 5000
    
    
    avg_reward = 0

    for episode in range(EpisodeNum):
        inputs_dict = {}
        state, _ = env.reset()
        total_reward = 0
        # print("Reset The Environment")
        while True:
            ouptuts = agent.inference(state)
            nstate, reward, done, truncted, info = env.step(ouptuts['action']['action'].item())
            total_reward += reward
            done = done or truncted

            agent.behavior_info_dict["advantage"].append(reward) 
            agent.behavior_info_dict["value"].append(ouptuts['value']) 
            agent.behavior_info_dict["done"].append(1 if done else 0) 
            if done :
                avg_reward+=total_reward
                if episode % 10 == 0:
                    print(f"{episode} reward is {avg_reward/10}")
                    avg_reward = 0
                break   
            agent.behavior_info_dict["action"].append(ouptuts['action']) 
            agent.behavior_info_dict["logits"].append(ouptuts['logits']) 
            agent.behavior_info_dict["decoder_mask"].append({"action": torch.ones(1)})          
            agent.input_dict['common_encoder'].append(torch.Tensor(state))
            state = nstate
        for i in range(1):
            agent.train()    
            agent.clean_data()
