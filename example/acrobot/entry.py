# builder = None
from hyurl.flow.drill_plugin.interface.flow_model import FlowModelPPO
from hyurl.flow.drill_plugin.interface.flow_env import FlowEnvImp as FlowEnvPPO
from example.acrobot.config import builder

flow_config = {
    'framework': "pytorch",
    'algorithm': {'flow_env': FlowEnvPPO, 'flow_model': FlowModelPPO},
    'builder': builder,
    'actor_config': {
        'actor_ac': {
            'training_models': [
                {
                    'model_name': 'acrobot_model',
                    'fragment_size': 512,
                    'replay_size': 1,
                    'sample_batch_size': 4096,
                    'max_data_reuse': 0,
                    'sync': False,
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
