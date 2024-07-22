import random
from re import M
import stat
from turtle import st
import torch 
import numpy  as np

from src.tools.gae import calculate_gae

class Fragment:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.dones = []
        self.values = []
        self.masks = []
        self.logits = []
        self.log_probs = []
    
    def size(self):
        return len(self.states) - 1
    
    def store(self, *piece):
        state, action, reward, log_prob, mask, done, value, logits = piece
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.values.append(value)
        # self.log_probs.append(log_prob)
        self.logits.append(logits)
        
    def gens(self):
        advantages = self.gae(self.rewards, self.values, self.dones)
        return self.states[:-1], self.actions[:-1], self.logits, self.masks[:-1], self.log_probs[:-1], self.values[:-1], self.dones[:-1], advantages

    # 计算广义优势估计 
    def gae(self, rewards, values, dones, gamma: float = 0.99, lamb: float = 0):
        ''' 
        GAE: 广义优势估计, 可以理解为:
                优化累积奖励[当前状态到游戏结束]的期望 <==> 优化累积优势[当前状态到游戏结束]的期望.
                奖励是绝对值, 而优势是相对值, 是平均情况的额外价值. 
                目的是减少估计的方差, 提高策略梯度算法的稳定性.
        gamma:
            折扣因子, 对未来的奖励进行折扣, 用于计算value, 评估当前状态的价值(未来折扣奖励期望)
            即便不采用GAE其实也需要gamma来计算累积折扣回报.     
        lamb:
            平滑因子, 平衡方差和偏差(lambda越大 偏差越小 方差越大)
            用于计算优势: 
                已知 s,a,r,s', 单步TD误差为 delta
                advantage 用于计算累积TD误差, 未来的TD误差乘 gamma, lamb
                lamb = 0 时, 相当于只计算单步的TD误差.
                lamb = 1 时, 相当于计算全部的TD误差. 
        '''
        # TD误差: 
        #   单步的 reward 减去 当前状态的单步奖励期望(两个状态的价值差) 相当于当前动作的优势
        #   gamma
        # advantage 是 多步TD误差
        # print(values)
        # values = values.detach().numpy()
        # print(values)
        # values = torch.Tensor(values)
        # values = torch.concat(values)
        advantage = 0.0
        advantages = []
        for i in reversed(range(len(rewards) - 1)):
            reward, value, next_value = rewards[i+1], values[i], values[i+1]
            non_terminate = 1 - int(dones[i + 1])
            delta = reward - (value - gamma * next_value * non_terminate)
            advantage = delta + gamma * lamb * advantage * non_terminate    
            advantages.append(advantage)
        return list(reversed(advantages))
        
class Memory:
    def __init__(self):
        self.reset()
        self.fragment = Fragment()
        self._size = 0
    def reset(self):
        self.fragment = Fragment()
        self.states = {}
        self.actions = {}
        self.logits = {}
        self.log_probs = {}
        self.masks = {}
        self.advantages = []
        self.is_terminals = []
        self.values = []
        self._size = 0 
    
    @property
    def size(self):
        return self._size
        
    def store(self, s, a, r, done, log_prob, mask, value, logit):
        self.fragment.store(s, a, r, log_prob, mask, done, value, logit)
        if done or len(self.fragment.actions) >= 128: 
            states, actions, logits, masks, logps, values,  dones, advs = self.fragment.gens()
            def trans(lis, dic):
                for ilis in lis:
                    for k, v in ilis.items():
                        if k not in dic:
                            dic[k] = []
                        dic[k].append(v)

            trans(states,   self.states)
            trans(actions,  self.actions)
            trans(logits,   self.logits)
            trans(masks,    self.masks)
            # trans(logps,    self.log_probs)
            
            self.values.extend(values)
            self.is_terminals.extend(dones)
            self.advantages.extend(advs)
            self._size = len(self.values)
            self.fragment = Fragment()

    def fstore(self, fragment:Fragment):
        states, actions, logits, masks, logps, values,  dones, advs = fragment.gens()
        def trans(lis, dic):
            for ilis in lis:
                for k, v in ilis.items():
                    if k not in dic:
                        dic[k] = []
                    dic[k].append(v)

        trans(states,   self.states)
        trans(actions,  self.actions)
        trans(logits,   self.logits)
        trans(masks,    self.masks)
        # trans(logps,    self.log_probs)
        
        self.values.extend(values)
        self.is_terminals.extend(dones)
        self.advantages.extend(advs)
        self._size = len(self.values)
        
    def get_batch(self, batch_size = 8):
        probabilities = torch.ones(len(self.values))  # 假设所有元素被选中的概率都相同
        batch_indices = torch.multinomial(probabilities, batch_size, replacement=False)

        return {
            'states':       { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in self.states.items() },
            'actions':      { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in self.actions.items() },
            'logits':       { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in self.logits.items() },
            'masks':        { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in self.masks.items() },
            # 'log_probs':    { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in self.log_probs.items() },
            'advantages':   torch.stack([torch.as_tensor(self.advantages[i])    for i in batch_indices]),
            'values':       torch.stack([torch.as_tensor(self.values[i])        for i in batch_indices]),
        }
