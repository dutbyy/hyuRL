from __future__ import annotations
from networkx import is_directed_acyclic_graph, topological_sort
from torch import nn
from torch.nn import functional as F
from typing import TYPE_CHECKING, Any, Dict, List, Callable, OrderedDict, Union
import networkx as nx
import torch

from . import Encoder, Decoder, Aggregator, ValueApproximator

HIDDEN_PREFIX = "__hidden_state_"
EMBEDING_PREFIX = "__embedding_"
LOGITS_PREFIX = "__logits_"
ACTION_PREFIX = "__action_"

DONE = "done"
LOGITS = "logits"
ACTION = "action"
VALUE = "value"
HIDDEN_STATE = "hidden_state"


is_class_dict = lambda class_dict: ("class" in class_dict) and ("params" in class_dict)


def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件"""

    if not is_class_dict(class_dict):
        raise ValueError(
            f"Expected a dict with keys 'class' and 'params', but got {class_dict}"
        )

    class_ = class_dict["class"]
    params = class_dict["params"]
    return class_(**params)


def construct_dag(config_dict, model_dict):
    # 有向无环图的网络, 可以依赖两类数据
    #   1. input_dict的数据
    #   2. 前置网络的输出数据
    dag = nx.DiGraph()  # 构造有向图
    for sub_model_name, sub_model_config in config_dict.items():
        sub_model = model_dict[sub_model_name]
        sub_model.name = sub_model_name
        inputs = sub_model_config.get("inputs", [])
        if not isinstance(inputs, List):
            inputs = [inputs]
        for source in inputs:
            dag.add_edge(source, sub_model_name)
    if not is_directed_acyclic_graph(dag):
        raise Exception("神经网络依赖异常: 网络配置有环")
    return dag, topological_sort(dag)


class ComplexNetwork(nn.Module):
    """
    模板化的神经网络

    Args:
        network_config (_type_): _description_
    """

    def __init__(self, network_config):
        super().__init__()
        self._network_config: Union[Dict] = network_config
        self.sub_model_dict = nn.ModuleDict(
            {
                name: construct(submodel_config)
                for name, submodel_config in self._network_config.items()
            }
        )
        for k, v in self.sub_model_dict.items():
            setattr(v, "__label__", f"{type(v).__name__}: {k}")

        dag, top_generator = construct_dag(self._network_config, self.sub_model_dict)
        self._dag = dag
        self.top_sorted = [it for it in top_generator]
        self._default_source_embeddings = torch.zeros(1)

    def forward(self, input_dict: dict, behavior_action_dict=None, training=False):
        state_dict = input_dict.copy()
        predict_output_dict = OrderedDict(
            {VALUE: None, LOGITS: {}, ACTION: {}, HIDDEN_STATE: {}}
        )
        for node_name in self.top_sorted:
            if node_name not in self.sub_model_dict:
                continue
            sub_model: Union[Encoder, Decoder, Aggregator, ValueApproximator] = (
                self.sub_model_dict[node_name]
            )
            sub_model_config = self._network_config[node_name]
            inputs = []
            for source in sub_model_config.get("inputs", []):
                if state_dict and source in state_dict:
                    inputs.append(state_dict[source])
            # inputs = [ state_dict[source] for source in sub_model_config.get('inputs', []) ]
            if len(inputs) == 0:
                continue  # raise ValueError("Model have not inputs")
            if isinstance(sub_model, Encoder):
                # inputs = torch.concat(inputs, -1)
                inputs = inputs[0]
                outputs, embeddings = sub_model(inputs, training)
                state_dict[node_name] = outputs
                state_dict[EMBEDING_PREFIX + node_name] = embeddings

            elif isinstance(sub_model, Aggregator):
                inputs = torch.concat(inputs, -1)
                hidden_state = state_dict.get(HIDDEN_PREFIX + node_name, None)
                episode_done = state_dict.get(DONE, None)
                if hidden_state is not None and episode_done is not None:
                    hidden_state = hidden_state * (
                        1 - torch.unsqueeze(episode_done, axis=-1)
                    )
                output, output_hidden_state = sub_model(
                    inputs, initial_state=hidden_state, training=training
                )
                if output_hidden_state is not None:
                    predict_output_dict[HIDDEN_STATE][node_name] = output_hidden_state
                    # print(hidden_state)
                state_dict[node_name] = output

            elif isinstance(sub_model, ValueApproximator):
                inputs = torch.concat(inputs, -1)
                value = sub_model(inputs)
                predict_output_dict[VALUE] = value

            elif isinstance(sub_model, Decoder):
                inputs = torch.concat(inputs, -1)
                source_encoder_name = sub_model_config.get("source_encoder_name", None)
                if source_encoder_name:
                    source_embeddings = state_dict[
                        EMBEDING_PREFIX + source_encoder_name
                    ]
                else:
                    source_embeddings = state_dict.get(
                        "default_source_embeddings", self._default_source_embeddings
                    )
                mask_config = sub_model_config.get("mask", None)
                if not mask_config:
                    mask = None
                elif isinstance(mask_config, str):
                    mask = state_dict[mask_config]
                else:
                    raise ValueError(f"Unknown mask type : {type(mask_config)}")
                behavior_action = (
                    behavior_action_dict[node_name] if behavior_action_dict else None
                )
                logits, action, embeddings = sub_model(
                    [inputs, source_embeddings],
                    action_mask=mask,
                    behavior_action=behavior_action,
                )
                state_dict[node_name] = embeddings
                state_dict[LOGITS_PREFIX + node_name] = logits
                state_dict[ACTION_PREFIX + node_name] = action
                predict_output_dict[LOGITS][node_name] = logits
                predict_output_dict[ACTION][node_name] = action
            else:
                raise Exception(f"Unsupport Submodel : {type(sub_model)}")
        return predict_output_dict

    def get_aggregator_init_state(self):
        if callable(getattr(self._aggregator, "get_initial_state")):
            return self._aggregator.get_initial_state()
        return None

    def log_probs(self, logits_dict, action_dict, decoder_mask):
        """
        J(θ)关于θ的梯度, 等价于 logp * R的梯度在πθ下的期望
        """
        log_prob_dict = {}
        for action_name, action in action_dict.items():
            decoder = self.sub_model_dict[action_name]
            distribution = decoder.distribution(logits_dict[action_name])
            # action_mask = torch.as_tensor(decoder_mask[action_name].squeeze())
            logp = distribution.log_prob(action).squeeze()
            # logp = logp *  action_mask
            log_prob_dict[action_name] = logp
        return log_prob_dict

    def entropy(self, logits_dict, decoder_mask):
        """
        计算每个动作的概率分布的熵
        """
        entropy_dict = {}
        for action_name, logits in logits_dict.items():
            decoder = self.sub_model_dict[action_name]
            distribution = decoder.distribution(torch.as_tensor(logits))
            entropy_dict[action_name] = torch.squeeze(distribution.entropy())
        #             entropy_dict[action_name] = torch.squeeze(distribution.entropy()) * torch.as_tensor(decoder_mask[action_name])
        return entropy_dict

    def kl(self, logits_dict, other_logits_dict, decoder_mask):
        with torch.no_grad():
            kl_dict = {
                action_name: F.softmax(logits)
                * (
                    F.log_softmax(logits)
                    - F.log_softmax(other_logits_dict[action_name])
                )
                for action_name, logits in logits_dict.items()
            }
        return kl_dict

    # def _aggregate(
    #     self, aggregator, input_dict, encoder_output_dict, hidden_state_key, training
    # ):
    #     encoder_output_list = list(encoder_output_dict.values())
    #     hidden_state = input_dict.get(hidden_state_key, None)
    #     episode_done = input_dict.get(DONE, None)

    #     if hidden_state is not None and episode_done is not None:
    #         hidden_state = hidden_state * (1 - torch.expand_dims(episode_done, axis=-1))
    #         aggregator_output, aggregator_state = aggregator(
    #             encoder_output_list, initial_state=hidden_state, training=training
    #         )
    #     return aggregator_output, aggregator_state
