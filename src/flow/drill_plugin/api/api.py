from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ObsData:
    obs: Dict[str, Any]
    extra_info_dict: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""

@dataclass
class ActionData:
    action: Dict[str, np.ndarray]
    predict_output: Dict[str, np.ndarray]
    action_mask: Dict[str, np.ndarray] = field(default_factory=dict)    # action_head name to mask
    agent_name: str = ""


class Env(ABC):
    """训练仿真环境基类.     
    Args:
        env_id (int): 环境ID, 整个训练任务内的EnvID, 且会在进程死亡重启后自增.
        extra_info (dict): 额外信息, 可以用户在定义Env时指定, 同时会有一些和训练相关的节点信息.  
    """  
    @abstractmethod
    def __init__(self, env_id: int, extra_info: dict):
      
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """ 重置环境并返回初始的observation
        Returns:
            observation(object): 初始的observation
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, command_dict: Dict) -> Tuple:
        """ 环境执行command指令
        Args: 
            command_dict(dict): agent-command的字典; 由用户通过agent的a2c接口返回.
        Returns:
            observation(object): 环境当前的observation
            done(bool): episode是否结束
            info(dict): 其他有用的信息
        """
        raise NotImplementedError


class Agent(ABC):
    """ Agent可以认为就是一个代理单元(一个抽象意义的Agent) 
        多个Agent可以并行进行操作, 每个Agent的决策相对独立.
        Agent之间没有先后顺序. 
        Agent分为两部分:
            1. 需要调用神经网络模型的. 即一个Agent Instance 会对应一个Model; 
                多个[实现相同、不同]Agent Instance可以共享一个Model; 
            2. 不需要调用模型的, 基于BT/FSM等;  

    Args:
        env_id (int): 环境ID, 整个训练任务内的EnvID, 且会在进程死亡重启后自增.
        extra_info (dict): 额外信息, 可以用户在定义Agent时指定.
    """
    @abstractmethod
    def __init__(self, env_id: int, extra_info: Tuple[Dict, Any], use_model=True, inference=False):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """ 重置环境并返回初始的observation
        Returns:
            observation(object): 初始的observation
        """
        raise NotImplementedError

    @abstractmethod
    def o2s(self) -> Any:
        """ 重置环境并返回初始的observation
        Returns:
            observation(object): 初始的observation
        """
        raise NotImplementedError
    
    @abstractmethod
    def a2c(self) -> Any:
        """ 重置环境并返回初始的observation
        Returns:
            observation(object): 初始的observation
        """
        raise NotImplementedError

    def rule_tick(self) -> Any:
        """ 重置环境并返回初始的observation
        Returns:
            observation(object): 初始的observation
        """
        raise NotImplementedError


    @abstractmethod
    def tick(self, agent_obs: Any, extra_info: Dict) -> Tuple:
        super().__tick__()
        raise NotImplementedError