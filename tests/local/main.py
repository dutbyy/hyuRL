import torch
from typing import Tuple
from .config import get_model
from .env import CartPoleEnv
from .sampler import Sampler
from .trainer import PPOTrainer


def train(
    num_iterations: int = 100,
    episodes_per_iteration: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    device: str = "cpu",
    target_reward: float = 495.0,
) -> Tuple[torch.nn.Module, float]:
    print("=" * 50)
    print("开始训练 CartPole 环境")
    print(f"设备: {device}")
    print(f"迭代次数: {num_iterations}")
    print(f"每次迭代回合数: {episodes_per_iteration}")
    print("=" * 50)

    model = get_model()
    env = CartPoleEnv()
    sampler = Sampler(model, env, device)
    trainer = PPOTrainer(
        model,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        device=device,
    )

    best_reward = 0.0

    for iteration in range(num_iterations):
        transitions = sampler.collect(num_episodes=episodes_per_iteration)
        stats = sampler.get_statistics()

        train_stats = trainer.train_step(transitions, epochs=4)

        if stats["mean_reward"] > best_reward:
            best_reward = stats["mean_reward"]

        if (iteration + 1) % 10 == 0:
            print(f"\n--- 迭代 {iteration + 1}/{num_iterations} ---")
            print(f"平均奖励: {stats['mean_reward']:.2f}")
            print(f"最大奖励: {stats['max_reward']:.2f}")
            print(f"最小奖励: {stats['min_reward']:.2f}")
            print(f"最佳奖励: {best_reward:.2f}")
            print(f"策略损失: {train_stats.get('policy_loss', 0):.4f}")
            print(f"价值损失: {train_stats.get('value_loss', 0):.4f}")
            print(f"熵: {train_stats.get('entropy', 0):.4f}")

        if best_reward >= target_reward:
            print(f"\n达到目标奖励 {target_reward}！训练完成！")
            break

    env.close()
    print("\n训练完成！")

    return model, best_reward


def evaluate(model, num_episodes: int = 10, device: str = "cpu") -> float:
    print("\n" + "=" * 50)
    print("开始评估模型...")
    print("=" * 50)

    env = CartPoleEnv()
    episode_rewards = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            obs_dict = env.get_observation_dict(observation)
            with torch.no_grad():
                output = model(obs_dict, training=False)

            action_logits = output["logits"]["action"]
            action = torch.argmax(action_logits, dim=-1).item()

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    env.close()

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)

    print(f"\n评估回合数: {num_episodes}")
    print(f"平均奖励: {mean_reward:.2f}")
    print(f"最大奖励: {max_reward:.2f}")
    print(f"最小奖励: {min_reward:.2f}")

    return mean_reward


if __name__ == "__main__":
    model, best_reward = train(num_iterations=100, episodes_per_iteration=5, target_reward=495.0)
    evaluate(model, num_episodes=10)
