import random
import torch 

from tools.gae import calculate_gae
class Fragments:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.advantages = []
        self.dones = []
        self.values = []
        self.masks = []
        self.traning_data = []
        self.logits = []
    
    def store(self, piece):
        state, action, reward, done, value = piece
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def calc(self):
        self.gae()

    # 计算广义优势估计 
    def gae(self, rewards, values, dones, gamma: float = 0.99, lamb: float = 0.95):
        ''' 
        GAE: 广义优势估计, 可以理解为:
                优化累积奖励[当前状态到游戏结束]的期望 <==> 优化累积优势[当前状态到游戏结束]的期望.
                奖励是绝对值, 而优势是相对值, 是平均情况的额外价值. 
                目的是减少估计的方差, 提高策略梯度算法的稳定性.
        gamma:
            类似于折扣因子, 对未来的奖励进行折扣
        lamb:
            优 
        '''
        advantage = 0.0
        advantages = []
        for i in reversed(range(len(rewards) - 1)):
            reward = rewards[i + 1]
            value = values[i]
            next_value = values[i + 1]
            non_terminate = 1 - int(dones[i + 1])
            delta = reward - (value - gamma * next_value * non_terminate)   # 真实的reward  减去 原有的期望奖励差
            advantage = delta + gamma * lamb * advantage * non_terminate    # 优势 = diff + 差加上累积优势的期望
            advantages.append(advantage)
        return list(reversed(advantages))
        
class Memory:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.advantages = []
        self.dones = []
        self.values = []
        self.masks = []
        self.traning_data = []
        self.logits = []
    
    def store(self, s, a, r, d, logits, mask, value):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.logits.append(logits)
        self.values.append(value)
        self.masks.append(mask)
        self.trajectory = []
            
    
    def get_batch(self, batch_size = 8):
        batch_indices = list(range(len(self.states)))
        random.shuffle(batch_indices)
        
        batch_indices = batch_indices[:batch_size]

        return {
            'states': torch.stack([self.states[i] for i in batch_indices]),
            'actions': torch.stack([self.actions[i] for i in batch_indices]),
            'log_probs': torch.stack([self.log_probs[i] for i in batch_indices]),
            'rewards': torch.stack([self.rewards[i] for i in batch_indices]),
            'is_terminals': torch.stack([self.is_terminals[i] for i in batch_indices])
        }
