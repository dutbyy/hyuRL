from tests.local.config import builder
from hyurl.flow.drill_plugin.interface.flow_model import FlowModelPPO
from hyurl.flow.drill_plugin.interface.flow_env import FlowEnvImp as FlowEnvPPO

flow_config = {
    'framework': "pytorch",
    'algorithm': {'flow_env': FlowEnvPPO, 'flow_model': FlowModelPPO},
    'builder': builder,
    'actor_config': {
        'actor_cartpole': {
            'training_models': [
                {
                    'model_name': 'cp_model',
                    'fragment_size': 512,
                    'replay_size': 1,
                    'sample_batch_size': 4096,
                    'sync': False,
                    'max_data_reuse': 0,
                    'putback_replays': False,
                    'sample_mode': "LIFO",
                    'replay_buffer_size': 1,
                },
            ],
            'inference_models': None,
            'episode_mode': False,
            'env_num': 40,
            'extra_info': {'index': 'training', 'description': 'used for training'},
        },
    },
}
