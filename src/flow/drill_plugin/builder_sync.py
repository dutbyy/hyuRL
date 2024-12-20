import os
from copy import deepcopy
from typing import Dict, List, Union

import numpy as np

import drill
from drill.env import Env
from drill.keys import CRITIC_HIDDEN_STATE, HIDDEN_STATE
from drill.model import Model
from drill.pipeline import GlobalPipeline
from drill.pipeline.pipeline_manager import PipelineManager
from drill.utils import construct


class Builder:
    """用于创建 `CommanderAgent`, `Env`, `Pipeline` 的构建器
    """

    @property
    def backend(self) -> str:
        """使用的后端，tensorflow or pytorch

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

    def build_env(self, env_id: int, extra_info) -> Env:
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

    def build_model(self, model_name: str) -> Model:
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

    def build_pipeline(self) -> PipelineManager:
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


class BuilderSync(Builder):

    def __init__(self,
                 agents: Dict[str, Dict[str, str]],
                 models,
                 env,
                 pipeline=None,
                 backend="pytorch"):
        self._agents = agents
        self._models = models
        self._env = env
        self._pipeline = pipeline
        self._save_params = {}
        self._learn_step = 0
        self._backend = backend

    def get_model_name(self, agent_name: str) -> str:
        return self._agents[agent_name]["model"]

    @property
    def backend(self):
        return self._backend

    @property
    def model_names(self):
        return list(self._models.keys())

    @property
    def agent_names(self):
        return list(self._agents.keys())

    @property
    def save_params(self):
        return self._save_params

    @property
    def learn_step(self):
        return self._learn_step

    def _get_agent_name(self, model_name: str):
        for name, agent in self._agents.items():
            if model_name == agent["model"]:
                return name
        raise ValueError(f"{model_name} does not exist. Please double-check the config.")

    def build_model(self, model_name: str, show_graph=False):
        model_config = self._models[model_name]
        network_config = model_config["params"]["network"]
        network = construct(network_config)
        pipeline = self.build_pipeline()
        agent_name = self._get_agent_name(model_name)
        model_config_copy = deepcopy(model_config)
        model_config_copy["params"]["network"] = network
        model = construct(model_config_copy)
        return model

    def build_env(self, env_id: int, extra_info: dict):
        self._env["params"]["env_id"] = env_id
        self._env["params"]["extra_info"] = extra_info
        env = construct(self._env)
        return env

    def build_pipeline(self) -> PipelineManager:
        class A :
            def reset(self): 
                pass
        return A()

    def get_initial_state(self, agent_name):
        return None
        