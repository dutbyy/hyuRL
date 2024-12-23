from __future__ import annotations
from typing import Dict, Tuple, Any, Union, List
from copy import deepcopy
import numpy as np

from hyuRL.src.flow.drill_plugin.api.flow_api import Builder, Model

def construct(class_dict: Dict):
    print(class_dict)
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
        # network_config = model_config["params"]["network"]
        # network = construct(network_config)
        model_config_copy = deepcopy(model_config)
        # model_config_copy["params"]["network"] = network
        model = construct(model_config_copy)
        return model

    def build_pipeline(self):
        from drill.pipeline import GlobalPipeline
        from drill.pipeline.pipeline_manager import PipelineManager
        # pipeline_ = construct(self._pipeline)
        ap_info = {}
        for name, setup in self._agents.items():
            ap_info[name] = self._pipeline[setup['pipeline']]

        template = {"class": GlobalPipeline, "params": {}}
        global_info = self._pipeline.get('global', template)
        history_len = self._pipeline.get('history_len', 0)
        pipeline_ = PipelineManager(ap_info, construct(global_info), history_len)
        return pipeline_
        
    def get_initial_state(self, agent_name):
        # from drill.model.tf.network.aggregator import DenseAggregator, GRUAggregator, LSTMAggregator
        # from drill.model.tf.network.commander import CommanderNetwork
        # model_name = self.get_model_name(agent_name)
        # model_config = self._models[model_name]
        # network_config = model_config["params"]["network"]
        # if issubclass(network_config["class"], CommanderNetwork):
        #     aggregator_config = network_config["params"]["aggregator_config"]
        #     aggregator_class = aggregator_config["class"]
        #     if aggregator_class == GRUAggregator:
        #         hidden_state = np.zeros(aggregator_config["params"]["state_size"], dtype=np.float32)
        #     elif aggregator_class == LSTMAggregator:
        #         hidden_state = np.zeros(2 * aggregator_config["params"]["state_size"],
        #                                 dtype=np.float32)
        #     elif aggregator_class == DenseAggregator:
        #         hidden_state = None
        #     else:
        #         raise NotImplementedError(
        #             f"Found that you used a custom aggregator class {aggregator_class}, "
        #             "you need inherit `BPBuilder` class and then overload the `get_initial_state` method. "
        #             "One thing to note is that the initial state returned should not contain the batch size dimension."
        #         )
        # else:
        #     raise NotImplementedError(
        #         f"Found that you did not use `CommanderNetwork`, "
        #         "you need inherit `BPBuilder` class and then overload the `get_initial_state` method. "
        #         "One thing to note is that the initial state returned should not contain the batch size dimension."
        #     )
        hidden_state_dict = {}
        # if hidden_state is not None:
        #     hidden_state_dict[HIDDEN_STATE] = hidden_state
        #     if not network_config["params"].get("share_critic", True):
        #         hidden_state_dict[CRITIC_HIDDEN_STATE] = hidden_state
        return hidden_state_dict
