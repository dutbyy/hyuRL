# builder = None
from example.atari.config import builder
from hyurl.flow.drill_plugin.interface.flow_model import FlowModelPPO
from hyurl.flow.drill_plugin.interface.flow_env import FlowEnvImp as FlowEnvPPO

flow_config = {
    'framework': "pytorch",
    'algorithm': {'flow_env': FlowEnvPPO, 'flow_model': FlowModelPPO},
    'builder': builder,
    'actor_config': {
            'actor_atari': {
                'training_models': [
                    {
                        'model_name': 'atari_model',
                        'fragment_size': 257,
                        'replay_size': 256,
                        'sample_batch_size': 32,
                        'max_data_reuse': 1,
                        'putback_replays': False,
                        'sample_mode': "LIFO",
                    },
                ],
                'inference_models': None,
                'episode_mode': False,
                'env_num': 32,
                'extra_info': {'index': 'training', 'description': 'used for training'},
            },
        },
    }
