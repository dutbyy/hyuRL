from __future__ import annotations
import networkx as nx
import torch
from networkx import is_directed_acyclic_graph, topological_sort
from torch import Value, nn
from torch.nn import functional as F
from typing import TYPE_CHECKING, Any, Dict, List, Callable, OrderedDict, Union
from hyuRL.src.tools.common import timer_decorator

HIDDEN_PREFIX = "__hidden_state_"
EMBEDING_PREFIX = "__embedding_"
LOGITS_PREFIX = "__logits_"
ACTION_PREFIX = "__action_"

DONE = "done"
LOGITS = "logits"
ACTION = "action"
VALUE = "value"
HIDDEN_STATE = "hidden_state"



from hyuRL.src.api.net.net import CommanderNetworkConfig
from hyuRL.src.api.net.net import CommonEncoderConfig, EntityEncoderConfig, SpatialEncoderConfig
from hyuRL.src.api.net.net import CategoricalDecoderConfig, GaussianDecoderConfig, SingleSelectiveDecoderConfig

from hyuRL.src.network import Encoder, Decoder, Aggregator, ValueApproximator
from hyuRL.src.network import CommonEncoder, EntityEncoder, SpatialEncoder
from hyuRL.src.network import CategoricalDecoder, GaussianDecoder, SingleSelectiveDecoder
from hyuRL.src.network import DenseAggregator
from hyuRL.src.network import ValueApproximator


def generate(config):

    if not isinstance(config, CommanderNetworkConfig):
        raise Exception("You need provice a config use class : [CommanderNetworkConfig]")
    DefaultFeatureLength = 256
    module_dict = {}
    config.quick_dict = {}
    for encoder_cfg in config.encoders:
        name = encoder_cfg.get_name()

        if isinstance(encoder_cfg, CommonEncoderConfig):
            encoder = CommonEncoder(
                in_features=encoder_cfg.feature_size,
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes,
                output_size=DefaultFeatureLength
            )
        elif isinstance(encoder_cfg, EntityEncoderConfig):
            encoder = EntityEncoder(
                length=encoder_cfg.length,
                in_features=encoder_cfg.feature_size,
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes,
                output_size=DefaultFeatureLength,
                transformer=encoder_cfg.transformer,
                pooling=encoder_cfg.pooling,
            )
        elif isinstance(encoder_cfg, SpatialEncoderConfig):
            encoder = SpatialEncoder(
                in_shape=encoder_cfg.shape,
                in_features= encoder_cfg.feature_size,
                channel_num= encoder_cfg.channel_num,
                output_size= DefaultFeatureLength,
                down_samples= encoder_cfg.down_samples,
                res_block_num= encoder_cfg.res_block_num,
            )
        else:
            raise Exception(f"Not Supported Encoder : {type(encoder_cfg)}")
        encoder_cfg.dependency = [encoder_cfg.feature_set.name]
        module_dict[name] = encoder
        config.quick_dict[name] = encoder_cfg


    aggregator = DenseAggregator(
        in_features = DefaultFeatureLength * len(config.encoders),
        hidden_layer_sizes = config.aggregator.hidden_layer_sizes,
        output_size = DefaultFeatureLength
    )
    name = config.aggregator.get_name()
    module_dict[name] = aggregator
    config.aggregator.dependency = [encoder.get_name() for encoder in config.encoders]
    config.quick_dict[name] = config.aggregator

    value = ValueApproximator(
        in_features = DefaultFeatureLength,
        hidden_layer_sizes = config.value.hidden_layer_sizes,
    )
    name = config.value.get_name()
    module_dict[name] = value
    config.value.dependency = [config.aggregator.get_name()]
    config.quick_dict[name] = config.value


    for decoder_cfg in config.decoders:
        name = decoder_cfg.get_name()

        if isinstance(decoder_cfg, CategoricalDecoderConfig):
            decoder = CategoricalDecoder(
                n = decoder_cfg.n,
                in_features = DefaultFeatureLength,
                hidden_layer_sizes=encoder_cfg.hidden_layer_sizes)
        elif isinstance(decoder_cfg, GaussianDecoderConfig):
            decoder = GaussianDecoder(
                n = decoder_cfg.n,
                in_features = DefaultFeatureLength,
                hidden_layer_sizes=decoder_cfg.hidden_layer_sizes,
            )
        elif isinstance(decoder_cfg, SingleSelectiveDecoderConfig):
            decoder = SingleSelectiveDecoder(
                in_features = DefaultFeatureLength,
                attention_size = decoder_cfg.attention_size,
            )
        else:
            raise Exception("Not Supported Encoder")
        module_dict[name] = decoder
        if not decoder_cfg.dependency:
            decoder_cfg.dependency = [config.aggregator.get_name()]
        config.quick_dict[name] = decoder_cfg

    return module_dict

CommanderNetworkConfig.generate = lambda self: generate(self)

def construct_dag(config_dict, model_dict):
    dag = nx.DiGraph()  # 构造有向图
    for net_cfg in config_dict.encoders + config_dict.decoders + [config_dict.aggregator] + [config_dict.value]:
        sub_model = model_dict[net_cfg.get_name()]
        sub_model.name = net_cfg.get_name()
        dependency = net_cfg.dependency
        if not isinstance(dependency, List):
            dependency = [dependency]

        for depend_name in dependency:
            dag.add_edge(depend_name, net_cfg.get_name())
    if not is_directed_acyclic_graph(dag):
        raise Exception("神经网络依赖异常: 网络配置有环")
    return dag, topological_sort(dag)


# 初始化网络参数
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Xavier初始化
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Xavier初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ComplexNetwork(nn.Module):
    """
    模板化的神经网络

    Args:
        network_config (CommanderNetworkConfig): 基于Commander的神经网络配置
    """

    def __init__(self, network_config: CommanderNetworkConfig):
        super().__init__()
        self._network_config: CommanderNetworkConfig = network_config
        self.sub_model_dict = nn.ModuleDict(
            self._network_config.generate()
        )
        for k, v in self.sub_model_dict.items():
            setattr(v, "__label__", f"{type(v).__name__}: {k}")

        dag, top_generator = construct_dag(self._network_config, self.sub_model_dict)
        self._dag = dag
        self.top_sorted = [it for it in top_generator]
        self._default_source_embeddings = torch.zeros(1)
        init_weights(self)
        print("init weight by function: xavier uniform")


    # @timer_decorator
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
            sub_model_config = self._network_config.quick_dict.get(node_name)
            inputs = []
            for source in sub_model_config.dependency:
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
                source_encoder_name = sub_model_config.source_encoder_name
                if source_encoder_name:
                    source_embeddings = state_dict[
                        EMBEDING_PREFIX + source_encoder_name
                    ]
                else:
                    source_embeddings = state_dict.get(
                        "default_source_embeddings", self._default_source_embeddings
                    )
                mask_config = sub_model_config.mask
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
        log_prob_dict = {}
        for action_name, action in action_dict.items():
            decoder = self.sub_model_dict[action_name]
            distribution = decoder.distribution(logits_dict[action_name])
            action_mask = torch.as_tensor(decoder_mask[action_name].squeeze())
            logp = distribution.log_prob(action).squeeze()
            logp = logp *  action_mask
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
