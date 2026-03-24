import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from .sampler import Sampler, Transition


class PPOTrainer:
    def __init__(
        self,
        model,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(advantages), np.array(returns)

    def train_step(self, transitions: List[Transition], epochs: int = 4) -> Dict[str, float]:
        if not transitions:
            return {}

        observations = np.array([t.observation for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions])
        old_logits = {k: torch.stack([t.logits[k] for t in transitions]) for k in transitions[0].logits}

        with torch.no_grad():
            last_obs = transitions[-1].next_observation
            last_obs_dict = {"observation": torch.from_numpy(last_obs).float().unsqueeze(0)}
            last_output = self.model(last_obs_dict, training=False)
            next_value = last_output["value"].item()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.from_numpy(advantages).float()
        returns = torch.from_numpy(returns).float()

        observations_tensor = torch.from_numpy(observations).float()
        actions_tensor = torch.from_numpy(actions).long()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(epochs):
            obs_dict = {"observation": observations_tensor}
            output = self.model(obs_dict, training=True)

            action_logits = output["logits"]["action"]
            action_probs = F.softmax(action_logits, dim=-1)
            action_log_probs = F.log_softmax(action_logits, dim=-1)
            selected_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                old_action_log_probs = F.log_softmax(old_logits["action"], dim=-1)
                old_selected_log_probs = old_action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(selected_log_probs - old_selected_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = output["value"].squeeze()
            value_loss = F.mse_loss(value_pred, returns)

            entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs,
            "entropy": total_entropy / epochs,
        }
