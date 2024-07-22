from multiprocessing import Process
import torch

from local.sampler import Sampler
from local.trainer import LocalTrainer
from local.predictor import serve
from src.algo.PPOPolicy import PPOPolicy
from src.memory.buffer import Memory
from local.predictor import PredictorClient
import time

def predictor_server(network_cfg):
    import asyncio
    from src.network import ComplexNetwork
    print("prepate to server")
    model = ComplexNetwork(network_cfg) 
    model.eval()
    model.to('cuda')
    asyncio.run(serve(model))
     

class Controller:
    def __init__(self) -> None:
        self.batch_size = 1080
    
    def print_info(self, train_step=0):
        print(f"trainning {train_step}")
        pass

    def train_run(self, policy_config):
    

        self.predict_server = Process(target=predictor_server, args=(policy_config, ))
        # self.predict_server.daemon = True
        self.predict_server.start()
        self.predict_server.join()
           
    def generate_data(self):
        self.sampler.restore_sampling()
        fragements = self.sampler.get_batch(self.batch_size)
        buffer = Memory()
        import timeit
        print("prepare for buffer")
        atime = timeit.default_timer() * 1000
        for frg in fragements:
            buffer.fstore(frg)
        btime = timeit.default_timer() * 1000
        print(f'make buffer eplased time {round(btime - atime, 2)} ms')
        train_data = buffer.get_batch(self.batch_size)
        
        return train_data
            
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    from src.network import *
    network_cfg = {
        "encoder_demo" : {
            "class": CommonEncoder,
            "params": {
                "in_features" : 4, 
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ['feature_a']
        },
        "aggregator": {
            "class": DenseAggregator,
            "params": {
                "in_features" : 128, 
                "hidden_layer_sizes": [256],
                "output_size": 256
            },
            "inputs": ['encoder_demo']
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features" : 256, 
                "hidden_layer_sizes": [64],
            },
            "inputs": ['aggregator']
        },
        "action" : {
            "class": CategoricalDecoder,
            "params": {
                "n" : 2,
                "hidden_layer_sizes": [4],
            },
            "inputs": ['aggregator']
        }
    }
    controller = Controller()
    controller.train_run(network_cfg)
    
    