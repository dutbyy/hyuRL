import torch

from local.predictor import serve

def predictor_server(network_config):
    import asyncio
    from src.network import ComplexNetwork
    from src.algo.PPOPolicy import PPOPolicy
    model = PPOPolicy(network_config, False, 'cpu')
    asyncio.run(serve(model))


if __name__ == '__main__':

    from src.network import *
    from local.net import network_cfg
    # predictor_server(network_cfg)
    import cProfile
    cProfile.run('predictor_server(network_cfg)','restats')


