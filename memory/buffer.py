import random
from re import M
import stat
from turtle import st
import torch 

from tools.gae import calculate_gae

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
    
    def store(self, piece):
        state, action, reward, log_prob, mask, done, value, logits = piece
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.logits.append(logits)
        
    def gens(self):
        advantages = self.gae(self.rewards, self.values, self.dones)
        return self.states[:-1], self.actions[:-1], self.logits[:-1], self.masks[:-1], self.log_probs[:-1], self.values[:-1], self.dones[:-1], advantages

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
        values = torch.Tensor(values)

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
        self.fragment = Fragment()
        self.size = 0
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
        self.size = 0 
        
    def store(self, s, a, r, done, log_prob, mask, value, logit):
        self.fragment.store([s, a, r, log_prob, mask, done, value, logit])
        if done or len(self.fragment.actions) >= 100: 
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
            trans(logps,    self.log_probs)
            
            self.values.extend(values)
            self.is_terminals.extend(dones)
            self.advantages.extend(advs)
            self.size = len(self.values)
            self.fragment = Fragment()

    
    def get_batch(self, batch_size = 8):
        probabilities = torch.ones(len(self.values))  # 假设所有元素被选中的概率都相同
        batch_indices = torch.multinomial(probabilities, batch_size, replacement=False)

        return {
            'states':       { k: torch.stack( [v[i] for i in batch_indices]) for k , v in self.states.items() },
            'actions':      { k: torch.stack( [v[i] for i in batch_indices]) for k , v in self.actions.items() },
            'logits':       { k: torch.stack( [v[i] for i in batch_indices]) for k , v in self.logits.items() },
            'masks':        { k: torch.stack( [v[i] for i in batch_indices]) for k , v in self.masks.items() },
            'log_probs':    { k: torch.stack( [v[i] for i in batch_indices]) for k , v in self.log_probs.items() },
            'advantages':   torch.stack([self.advantages[i] for i in batch_indices]),
            'values':       torch.stack([self.values[i] for i in batch_indices]),
        }
