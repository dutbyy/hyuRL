# builder = None
from hyuRL.example.atari.config import builder
from hyuRL.src.flow.drill_plugin.interface.flow_model import FlowModelPPO
from hyuRL.src.flow.drill_plugin.interface.flow_env import FlowEnvImp as FlowEnvPPO

flow_config = {
    'framework': "pytorch",
    'algorithm': {'flow_env': FlowEnvPPO, 'flow_model': FlowModelPPO},
    'builder': builder,
    'actor_config': {
        'actor_atari': {
            'training_models': [
                {
                    'model_name': 'atari_model',
                    'fragment_size': 256,
                    'replay_size': 1,
                    'sample_batch_size': 1024,
                    'max_data_reuse': 2,
                    'putback_replays': False,
                    'sample_mode': "LIFO",
                    'replay_buffer_size': 16,
                },
            ],
            'inference_models': None,
            'episode_mode': False,
            'env_num': 40,
            'extra_info': {'index': 'training', 'description': 'used for training'},
        },
    },
}
