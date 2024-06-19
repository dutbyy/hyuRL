import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

# 超参数
learning_rate = 0.0003
gamma = 0.99
clip_epsilon = 0.2
k_epochs = 4
update_timestep = 4000
max_episodes = 3000
eps_clip = 0.2

# 创建环境
env = gym.make('CartPole-v1')

# 策略网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)  # 确保动作是长整型
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, torch.squeeze(state_values), dist_entropy

# PPO算法
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.tensor(memory.states).float()
        old_actions = torch.tensor(memory.actions).long()  # 确保动作是长整型
        old_logprobs = torch.tensor(memory.logprobs).float()

        for _ in range(k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

# 存储交互数据
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# 训练主函数
if __name__ == "__main__":
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    memory = Memory()
    ppo = PPO(state_dim, action_dim)

    timestep = 0
    running_reward = 0
    avg_length = 0

    for episode in range(max_episodes):
        state,_ = env.reset()
        episode_reward = 0
        for t in range(1, 10000):
            timestep += 1
            action, action_logprob = ppo.policy_old.act(state)
            state, reward, done, __, _ = env.step(action)

            memory.states.append(state)
            memory.actions.append(action)  # 确保动作是整数
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break

        running_reward += episode_reward
        avg_length += t

        if episode % 10 == 0:
            avg_length = int(avg_length / 10)
            running_reward = int(running_reward / 10)
            print(f"Episode {episode}, Avg length: {avg_length}, Avg reward: {running_reward}")
            running_reward = 0
            avg_length = 0

    env.close()
