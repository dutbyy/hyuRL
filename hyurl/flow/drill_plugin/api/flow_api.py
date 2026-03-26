"""
仿照Flow文档 http://11.1.203.3:8787/flow 提供的的一套Api抽象类和数据类
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union
from numpy.typing import NDArray
import numpy as np

NestedNDArray = Union[NDArray, list[Any], dict[str, Any]]


@dataclass
class EnvironmentDescriptor:
    node_id: int
    num_envs_on_this_node: int
    num_envs_on_this_actor: int
    environment_id_on_this_node: int
    environment_id_on_this_actor: int
    environment_id_on_this_task: int
    environment_creator_user_args: dict


# Drill-限制
class Builder:
    """用于创建 `CommanderAgent`, `Env`, `Pipeline` 的构建器"""

    @property
    def backend(self) -> str:
        """使用的后端, tensorflow or pytorch

        Returns
        -------
        str
            "tensorflow"/"pytorch"

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    def save_params(self) -> dict:
        """保存模型相关的参数

        Returns
        -------
        dict
            example: {
                "model_name": {
                    "path": "ckpts",
                    "interval": 100,
                    "mode": "npz"
                }
            }

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    def learn_step(self) -> int:
        """用于给 model.learn_step 设置初始值。

        Returns
        -------
        int

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def get_model_name(self, agent_name: str) -> str:
        """查找给定 agent 对应的 model

        Parameters
        ----------
        agent_name : str
            agent 的名字

        Returns
        -------
        str
            model 的名字
        """
        raise NotImplementedError

    @property
    def model_names(self) -> List[str]:
        """获取所有 model 的名字

        Returns
        -------
        List[str]
            所有 model 的名字
        """
        raise NotImplementedError

    @property
    def agent_names(self) -> List[str]:
        """获取所有 agent 的名字

        Returns
        -------
        List[str]
            所有 agent 的名字
        """
        raise NotImplementedError

    def build_env(self, env_id: int, extra_info):
        """构建 env

        Parameters
        ----------
        env_id : int
            用于 env 实例，通常 env 会根据不同的 env_id 设置不同的 random seed。

        Returns
        -------
        Env
            见 `drill.Env`
        """
        raise NotImplementedError

    def build_model(self, model_name: str):
        """构建 model

        Parameters
        ----------
        model_name : str
            model 的名字

        Returns
        -------
        Model
            见 `Model`
        """
        raise NotImplementedError

    def build_pipeline(self):
        """构建 pipeline

        Returns
        -------
        Pipeline
            见 `Pipeline`
        """
        raise NotImplementedError

    def get_initial_state(self, agent_name: str) -> Union[np.ndarray, None]:
        """获取给定 agent 的 initial state。当网络中有 rnn 时，需要给出 initial state，默认是 None

        Parameters
        ----------
        agent_name : str
            agent 的名字

        Returns
        -------
        Union[np.ndarray, None]
            当网络中有 rnn 时，返回 rnn 的 initial state；当网络中没有 rnn 时，返回 None
        """
        return None


class Environment:
    """用户需要封装自己的Environment为符合上述行为的Class;
    如果需要与n个agent交互, 则会将agent编号为 0, 1, 2, …, n-1
    编号顺序在Actor中指定

    Args:
        environment_descriptor (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """

    def __init__(self, environment_descriptor: EnvironmentDescriptor):
        """创建一个新的环境, 用户需要提供environment_creator给Flow
        Flow会使用形如 environment_creator(environment_descriptor) 的调用方式创建Environment Object.
        因此如果此类型的__init__方法参数只有environment_descriptor, 可以直接将类型名作为environment_creator
        """
        raise NotImplementedError("This method is not yet implemented")

    def reset(self) -> None:
        """开始一个新的episode"""
        raise NotImplementedError("This method is not yet implemented")

    def observe(self) -> Dict[str, NestedNDArray]:
        """获取agent的观测数据和模型名称

        Returns:
            obs (Dict): 所有agent的观测数据和模型名称

        Example:
            e.g. `return {agent_name: {"obs": obs, "model": model_name, "timeout_ms": timeout_ms}}`
            timeout_ms 可以不填写
        """
        raise NotImplementedError("This method is not yet implemented")

    def step(agent_name: str, action: NestedNDArray) -> NestedNDArray:
        """action内的所有np.array的生命周期, 仅在step函数的调用期间, 保存数据需要deepcopy

        Args:
            agent_name (str): agent名称
            action (NestedNDArray): Model 预测结果
        Returns:
            third_place (NestedNDArray: 三元组第三个位置的占位符
                    PS: 这个位置按照一般RL定义的三元组 (s, a, r) 的第三个位置, 但实际可以理解就是一个占位符
        """
        raise NotImplementedError("This method is not yet implemented")

    def enhance_fragment(
        agent_name: str, fragments: List[Tuple[NDArray, NDArray, NDArray]]
    ) -> None:
        """用户需要原地更改输入的fragment, 并保证更改后的fragment类型 为list of enhanced transitions,
            其中enhanced transition的类型为nested np.array, 且其shape和dtype都必须固定
            可以认为 默认的fragments 是一系列的transitions
            transitions本身就是一个step的 s, a, r
            即 原始的
                1. agent2state (env.observe 返回值)
                2. agent2action (env.step 参数)
                3. agent2reward (env.step 返回值)
            其实一个transitions里面有什么数据不重要[PS:只要保证是nested-ndarray]
            fragment_size: 确定什么时候调用一次enhance_fragments
            transition: 训练数据的最小单元, replay_size决定连续性
                这个在model.learn的时候, 称为了piece?

        Args:
            agent_name (str): agent名称
            fragment (List[Tuple[NDArray, NDArray, NDArray]]): 长度在[1, fragment_size]范围内的list, 每个元素为长度为3的list[obs, action, reward]
        """
        raise NotImplementedError("This method is not yet implemented")

    def set_config(config):
        pass


class Model:
    def __init__(self, model_name: str, builder: Builder):
        raise NotImplementedError("This method is not yet implemented")

    def __getstate__(self):
        """返回可使用pickle序列化的Model State"""
        raise NotImplementedError("This method is not yet implemented")

    def setstate_learn(self, state: NestedNDArray):
        raise NotImplementedError("This method is not yet implemented")

    def setstate_predict(self, state: NestedNDArray):
        raise NotImplementedError("This method is not yet implemented")

    def learn(self, batch: List[Dict[str, NestedNDArray]]):
        raise NotImplementedError("This method is not yet implemented")

    def predict(self, batch_obs: Dict[str, NestedNDArray]):
        raise NotImplementedError("This method is not yet implemented")

    def get_weights(self):
        raise NotImplementedError("This method is not yet implemented")

    def set_weights(self, weights: NestedNDArray):
        raise NotImplementedError("This method is not yet implemented")

    def load_weights(self, model_path: str, backend: str, mode):
        raise NotImplementedError("This method is not yet implemented")

    def set_config(config):
        pass
