from typing import List


def calculate_gae(
    reward_list: List[float],
    value_estimates: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lamb: float = 0.95,
) -> List[float]:
    """计算广义优势估计 (Generalized Advantage Estimation, GAE)

    Parameters
    ----------
    reward_list : List[float]
        连续的奖励值, 表示在每个状态执行动作后获得的奖励。
    value_estimates : List[float]
        连续的状态值, 表示每个状态的估计值。
    dones : List[bool]
        连续的done标志, 表示每个状态是否是终止状态。
    gamma : float, optional
        折扣因子, 默认为0.99。
    lamb : float, optional
        GAE中的参数, 默认为0.95。

    Returns
    -------
    List[float]
        连续的优势值, 表示在每个状态执行动作的优势。
    """
    advantage = 0.0
    advantages = []
    for i in reversed(range(len(reward_list) - 1)):
        reward = reward_list[i + 1]
        value = value_estimates[i]
        next_value = value_estimates[i + 1]
        non_terminate = 1 - int(dones[i + 1])
        delta = reward + gamma * next_value * non_terminate - value
        advantage = delta + gamma * lamb * advantage * non_terminate
        advantages.append(advantage)
    return list(reversed(advantages))
