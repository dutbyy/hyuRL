from calendar import EPOCH
import torch
from torch import optim
import gym
from network.template import TemplateNetwork
from loss.ppo import PPOLoss

from tools.gae import calculate_gae
from memory.buffer import Memory
is_cuda = torch.cuda.is_available()
# is_cuda = False
device = torch.device("cuda" if is_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)
from tools.gpu import auto_move

class AgentPPO:
    def __init__(self, *args):
        self._network = TemplateNetwork(*args)
        self._network.to(device)
        self._optimizer = optim.Adam(self._network.parameters(), lr=1e-4)
        self._loss_fn = PPOLoss()
        self.memory = Memory()
        
    def inference(self, state):
        inputs = {"common_encoder": torch.Tensor(state)}
        inputs = auto_move(inputs)
        with torch.no_grad():
            outputs = self._network(inputs)
        action = outputs['action']
        return outputs

    def train(self):# , inputs_dict, behavior_info_dict):
        """
        训练的数据来源: 
            输入: 
                inputs:                 轨迹的状态
                behavior_info :         轨迹的采样
                    actions:            动作
                    logits:             动作的分布原始logits
                    mask:               动作的mask
            
            目的都是用于计算loss
            loss 的计算需要用到
                old_logp    : 历史动作的概率
                logp        : 当前网络动作概率
                advantage   : 优势
                value       : 价值
                target_value[behavior value]: 
                entropy
        """
        
        
        # 从behavior_info_dict中提取信息
        inputs_dict, behavior_info_dict =  self.memory.get_batch(16)
        
        behavior_action_dict = behavior_info_dict.get("action")
        behavior_logits_dict = behavior_info_dict.get("logits")
        behavior_mask_dict = behavior_info_dict.get("mask")
        advantage = behavior_info_dict.get("advantage")
        dones = behavior_info_dict.get("done")
    
        
        behavior_decoder_mask = {k : torch.stack(v) for k, v in behavior_decoder_mask.items() }
        behavior_decoder_mask = auto_move(behavior_decoder_mask)
        behavior_values = torch.Tensor(behavior_info_dict['value'])
        behavior_values = auto_move(behavior_values)
        advantages = torch.Tensor(calculate_gae(advantages, behavior_values, dones))
        advantage = auto_move(advantage)
        behavior_values = behavior_values[:-1]
        
        old_logp_dict = self._network.log_probs(behavior_logits_dict, behavior_action_dict, behavior_mask_dict)
        old_logp = sum(old_logp_dict.values())
        
        for _ in range(EPOCH):
            predict_output_dict = self._network(inputs_dict, behavior_action_dict, training=True)
            logits_dict = predict_output_dict['logits']
            
            logp_dict = self._network.log_probs(logits_dict, behavior_action_dict, behavior_mask_dict)
            logp = sum(logp_dict.values())
    
            # 计算当前策略的熵
            entropy_dict = self._network.entropy(logits_dict, behavior_decoder_mask)
            entropy = torch.mean(sum(entropy_dict.values()))
            
            value = predict_output_dict['value']
            value = auto_move(value)


            # 计算 target_value 参考：“A Closer Look At Deep Policy Gradients”
            # https://arxiv.org/pdf/1811.02553.pdf
            target_value = advantage + behavior_values 

            # logging.warning(f"value:        {value}             {value.shape}")
            # logging.warning(f"target_value: {target_value}      {target_value.shape}")
            # logging.warning(f"old_value:    {behavior_value}    {behavior_value.shape}")
            # logging.warning(f"advantage:    {advantage}         {advantage.shape}")
            # normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
            loss, policy_loss, value_loss, entropy_loss = self._loss_fn(
                    old_log_prob=old_logp,
                    log_prob=logp,
                    advantage=advantages,
                    old_value=behavior_values,
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
    
    agent = AgentPPO(encoder_config, aggregator_config, decoder_config, value_config)
    EpisodeNum = 5000
    
    avg_reward = 0

    for episode in range(EpisodeNum):
        inputs_dict = {}
        state, _ = env.reset()
        total_reward = 0
        # print("Reset The Environment")
        while True:
            outputs = agent.inference(state)
            nstate, reward, done, truncted, info = env.step(outputs['action']['action'].item())
            total_reward += reward
            done = done or truncted

            
            action_mask = {"action": torch.ones(1)}
            agent.memory.store(state, outputs['action'], reward, done, outputs['logits'], action_mask, outputs['value'])
            if done :
                avg_reward+=total_reward
                if episode % 10 == 0:
                    print(f"{episode} reward is {avg_reward/10}")
                    avg_reward = 0
                break   
            
            # agent.behavior_info_dict["advantage"].append(reward) 
            # agent.behavior_info_dict["value"].append(ouptuts['value']) 
            # agent.behavior_info_dict["done"].append(1 if done else 0) 
            # agent.behavior_info_dict["action"].append(ouptuts['action']) 
            # agent.behavior_info_dict["logits"].append(ouptuts['logits']) 
            # agent.behavior_info_dict["decoder_mask"].append({"action": torch.ones(1)})          
            # agent.input_dict['common_encoder'].append(torch.Tensor(state))
            state = nstate
        for i in range(1):
            agent.train()    
            # agent.clean_data()
