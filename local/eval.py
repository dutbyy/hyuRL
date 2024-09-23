from multiprocessing import Process
import torch

from local.sampler import Sampler
from local.trainer import LocalTrainer
from local.predictor import serve
from src.algo.PPOPolicy import PPOPolicy
from src.memory.buffer import Memory
from local.predictor import PredictorClient
import time

import pickle


class Controller:
    def __init__(self) -> None:
        self.batch_size = 1024 * 4

    def print_info(self, train_step=0):
        print(f"trainning {train_step}")
        pass

    def eval(self, policy_config):
        delayed_policy = {"class": PPOPolicy, "params": {"policy_config": policy_config, 'device': 'cpu' }}
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=1)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()


    def train_run(self, policy_config):


        # self.predict_server = Process(target=predictor_server, args=(policy_config, ))
        # # self.predict_server.daemon = True
        # self.predict_server.start()


        delayed_policy = {"class": PPOPolicy, "params": {"policy_config": policy_config, 'device': 'cpu' }}
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=4)

        delayed_policy = {"class": PPOPolicy, "params": {"policy_config": policy_config, 'device': 'cpu', "trainning": True}}
        self.trainer = LocalTrainer(delayed_policy=delayed_policy)
        self.predict_client = PredictorClient('localhost', 50051, False)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()
            self.trainer.train(train_data)
            # self.sampler.set_weight(self.trainer.get_state_dict())
            ret = self.predict_client.update_weight(self.trainer.get_state_dict())
            print(f"step: {train_step} update weight over ")

    def generate_data(self):
        fragements = self.sampler.get_batch(self.batch_size)
        buffer = Memory()
        import timeit
        #print("prepare for buffer")
        atime = timeit.default_timer() * 1000
        for frg in fragements:
            buffer.fstore(frg)
        btime = timeit.default_timer() * 1000
        #print(f'make buffer eplased time {round(btime - atime, 2)} ms')
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
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ['feature_a']
        },
        "aggregator": {
            "class": DenseAggregator,
            "params": {
                "in_features" : 128,
                "hidden_layer_sizes": [256, 128],
                "output_size": 256
            },
            "inputs": ['encoder_demo']
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features" : 256,
                "hidden_layer_sizes": [256, 256, 128],
            },
            "inputs": ['aggregator']
        },
        "action" : {
            "class": CategoricalDecoder,
            "params": {
                "n" : 2,
                "hidden_layer_sizes": [128, 128],
            },
            "inputs": ['aggregator']
        }
    }
    controller = Controller()
    controller.eval(network_cfg)



