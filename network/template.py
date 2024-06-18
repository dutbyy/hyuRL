import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Dict, List
from encoder.complex import ComplexEncoder

DONE = 'done'
LOGITS = 'logits'
ACTION = 'action'
VALUE = 'value'

def is_class_dict(class_dict: Dict):
    return ("class" in class_dict) and ("params" in class_dict)

def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件
    """

    if not is_class_dict(class_dict):
        raise ValueError(f"Expected a dict with keys 'class' and 'params', but got {class_dict}")

    class_ = class_dict["class"]
    params = class_dict["params"]
    return class_(**params)


class TemplateNetwork(nn.Module):
    ''' 
    模板化的神经网络
    '''
    
    def __init__(self, 
                    encoder_config,
                    aggregator_config,
                    decoder_config,
                    value_config,
                    share_critic = True):
        super().__init__()
        
        self._encoder_config = encoder_config       
        self._aggregator_config = aggregator_config       
        self._decoder_config = decoder_config       
        self._value_config = value_config       
        self._share_critic = share_critic       

        # self._encoders = {name: construct(encoder) for name, encoder in self._encoder_config.items()}
        self._complex = ComplexEncoder(self._encoder_config)
        self._aggregator = construct(self._aggregator_config)
        self._decoders = {name: construct(decoder) for name, decoder in self._decoder_config.items()}
        self._value = construct(self._value_config)
        # self._dependency = Dependency(self._encoder_config, self._encoders)
        self._default_source_embeddings = torch.zeros(1)

        # if not share_critic:
        #     self.critic_encoder_dict = {}
        #     for name, encoder in encoder_config.items():
        #         self.critic_encoder_dict[name] = construct(encoder)
        #     self._critic_dependency = Dependency(self._encoder_config, self.critic_encoder_dict)

        #     self._critic_aggregator = construct(aggregator_config)
     
    def forward(self, input_dict, behavior_action_dict = None, training = False):
        # Encode
        encoders_output_dict, encoder_embedding_dict = self._complex(input_dict, training)
        

        # Aggregate
        aggregator_output, aggregator_state = self._aggregate(self._aggregator, input_dict, encoders_output_dict, 256, training)
        
        # Decode
        decoder_embedding_dict, decoder_logits_dict, decoder_action_dict = {}, {}, {}
        for name, decoder in self._decoders.items():
            source_embeddings = encoder_embedding_dict[decoder.source_encoder_name] if decoder.source_encoder_name \
                                    else input_dict.get("default_source_embeddings", self._default_source_embeddings)

            dependent_decoder_name = decoder_config.get('dependency', None)
            dependency_embeddings = decoder_embedding_dict[dependent_decoder_name] if dependent_decoder_name else None
            inputs = dependency_embeddings if dependency_embeddings else aggregator_output
            mask_config = decoder_config.get('mask', None)
            
            if not mask_config:
                mask = None
            elif isinstance(mask_config, str):
                mask = input_dict[mask_config]
            elif isinstance(mask_config, Callable):
                mask = mask_config(input_dict, decoder_action_dict.get(dependent_decoder_name, None))
            else:
                raise ValueError(f"Unknown mask type : {type(mask_config)}")
        
            behavior_action = behavior_action_dict[name] if behavior_action_dict else None 
            logits, action, embeddings = decoder([inputs, source_embeddings], 
                                                    action_mask=mask, 
                                                    behavior_action=behavior_action)    
            decoder_embedding_dict[name] = embeddings
            decoder_logits_dict[name] = logits
            decoder_action_dict[name] = action
            
        # Predict Output
        if self._share_critic:
            value = self._value(aggregator_output)
        else:
            # TODO
            value = self._value(aggregator_output)
        
        predict_output_dict = {
            LOGITS: decoder_logits_dict,
            ACTION: decoder_action_dict,
            VALUE:  value,
        }

        if aggregator_state:
            predict_output_dict[HIDDEN_STATE] = aggregator_state 
        return predict_output_dict

    def get_aggregator_init_state(self):
        if callable(getattr(self._aggregator, "get_initial_state")):
            return self._aggregator.get_initial_state()
        return None

    def log_probs(self, logits_dict, action_dict, decoder_mask):
        '''  
        J(θ)关于θ的梯度, 等价于 logp * R的梯度在πθ下的期望 
        '''
        log_prob_dict = {}
        for action_name, decoder in self._decoder_dict.items():
            distribution = decoder.distribution(logits_dict[action_name])
            action = action_dict[action_name]
            action_mask = decoder_mask[action_name]
            logp = distribution.log_prob(action).squeeze() * action_mask
            log_prob_dict[action_name] = logp
        return log_prob_dict
    
    def entropy(self, logits_dict, decoder_mask):
        '''
        计算每个动作的概率分布的熵 
        '''
        entropy_dict = {}
        for action_name, decoder in self._decoder_dict.items():
            distribution = decoder.distribution(logits_dict[action_name])
            entropy_dict[action_name] = torch.squeeze(distribution.entropy()) * decoder_mask[action_name]
        return entropy_dict
    
    def kl(self, logits_dict, other_logits_dict, decoder_mask):
        kl_dict = {}
        for action_name, decoder in self._decoder_dict.items():
            distribution = decoder.distribution(logits_dict[action_name])
            other_distribution = decoder.distribution(other_logits_dict[action_name])
            kl_dict[action_name] = torch.squeeze(distribution.kl(other_distribution)) * decoder_mask[action_name]
        return kl_dict
    
    def _aggregate(self, aggregator, input_dict, encoder_output_dict, hidden_state_key, training):
        encoder_output_list = list(encoder_output_dict.values())
        hidden_state = input_dict.get(hidden_state_key, None)
        episode_done = input_dict.get(DONE, None)
        
        if hidden_state is not None and episode_done is not None :
            hidden_state = hidden_state * (1 - torch.expand_dims(episode_done, axis=-1))
        aggregator_output, aggregator_state = aggregator(encoder_output_list, 
                                                            initial_state = hidden_state,
                                                            training = training)
        return aggregator_output, aggregator_state
   
   
if __name__ == '__main__':
    from encoder.common import CommonEncoder
    from decoder.categorical import CategoricalDecoder
    from value import ValueApproximator
    from aggregator.dense import DenseAggregator
    
    encoder_config  = {
        "encoder_demo" : {
            "class": CommonEncoder,
            "params": {
                "in_features" : 128, 
                "hidden_layer_sizes": [256, 128],
            }
        } 
    }
    decoder_config = {
        "decoder_demo" : {
            "class": CategoricalDecoder,
            "params": {
                "n" : 5,
                "hidden_layer_sizes": [256, 128],
            }
        } 
    }
    aggregator_config = {
        "class": DenseAggregator,
        "params": {
            "in_features" : 128, 
            "hidden_layer_sizes": [256, 128],
            "output_size": 256,
        }
    } 
    value_config = {
       "class": ValueApproximator,
       "params": {
           "hidden_layer_sizes": [256, 128],
       }    
    }
    network = TemplateNetwork(encoder_config, aggregator_config, decoder_config, value_config)
    output = network({"encoder_demo": torch.rand(10, 128)})
    print(output['action'])