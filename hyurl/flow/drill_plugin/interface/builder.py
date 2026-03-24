from __future__ import annotations
from typing import Dict, Tuple, Any, Union, List
from copy import deepcopy
import numpy as np

from hyurl.flow.drill_plugin.api.flow_api import Builder, Model

def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件
    """
    is_class_dict = lambda class_dict : ("class" in class_dict) and ("params" in class_dict)
    if not is_class_dict(class_dict):
        raise ValueError(f"Expected a dict with keys 'class' and 'params', but got {class_dict}")

    class_ = class_dict["class"]
    params = class_dict["params"]
    return class_(**params)


class ExBuilder(Builder):
    def __init__(self,
                 agents: Dict[str, Dict[str, str]],
                 models,
                 env,
                 pipeline,
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

    def _get_agent_name(self, model_name: str):
        for name, agent in self._agents.items():
            if model_name == agent["model"]:
                return name
        raise ValueError(f"{model_name} does not exist. Please double-check the config.")

    @property
    def backend(self) -> str:
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

    def build_env(self, env_id: int, extra_info):
        self._env["params"]["env_id"] = env_id
        self._env["params"]["extra_info"] = extra_info
        env = construct(self._env)
        return env

    def build_model(self, model_name: str) -> Model:
        model_config = self._models[model_name]
        model_config_copy = deepcopy(model_config)
        model = construct(model_config_copy)
        self._save_params[model_name] = {
            'interval': model_config["save"].get('interval', 100),
            'mode': model_config["save"].get('mode', 'npz'),
            'path': '/job/model',
        }
        return model

    def build_pipeline(self):
        from drill.pipeline import GlobalPipeline
        from drill.pipeline.pipeline_manager import PipelineManager
        ap_info = {}
        for name, setup in self._agents.items():
            ap_info[name] = self._pipeline[setup['pipeline']]

        template = {"class": GlobalPipeline, "params": {}}
        global_info = self._pipeline.get('global', template)
        history_len = self._pipeline.get('history_len', 3)
        pipeline_ = PipelineManager(ap_info, construct(global_info), history_len)
        return pipeline_

    def get_initial_state(self, agent_name):
        hidden_state_dict = {}
        return hidden_state_dict
