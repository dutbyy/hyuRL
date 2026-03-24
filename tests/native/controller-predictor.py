import torch

from tests.native.predictor import serve

def predictor_server(network_config):
    import asyncio
    from hyurl.network import ComplexNetwork
    from hyurl.algo.PPOPolicy import PPOPolicy
    model = PPOPolicy(network_config, False, 'cpu')
    asyncio.run(serve(model))


if __name__ == '__main__':

    from hyurl.network import *
    from tests.native.net import network_cfg
    predictor_server(network_cfg)

