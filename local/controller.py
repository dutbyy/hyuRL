from multiprocessing import Process
from src.tools.common import timer_decorator
import torch
import logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)



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
        self.__init_logger()

    def __init_logger(self):
        self.logger = logging.getLogger(f"model_learn")
        self.logger.setLevel(10)
        if not self.logger.handlers:
            try:
                handler = logging.FileHandler(f"./model_learn.log")
            except:
                handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s'
                # '[%(name)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s'
            )
            handler.setFormatter(formatter)#logging.Formatter('[%(name)s] [%(asctime)s] [%(filename)s:%(lineno)s] %(message)s', datefmt='%y-%m-%d %H:%M:%S'))
            self.logger.addHandler(handler)
            handler =  logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info('Logger Init Finished')


    def eval(self, policy_config):
        delayed_policy = {"class": PPOPolicy, "params": {"network_config": policy_config, 'device': 'cpu' }}
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=1)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()

    @timer_decorator
    def train_run(self, policy_config):
        delayed_policy = {"class": PPOPolicy, "params": {"network_config": policy_config, 'device': 'cpu' }}
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=4)

        delayed_policy = {"class": PPOPolicy, "params": {"network_config": policy_config, 'device': 'cpu', "trainning": True}}
        self.trainer = LocalTrainer(delayed_policy=delayed_policy)
        self.predict_client = PredictorClient('localhost', 50051, False)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()
            eplased_times = self.trainer.train(train_data)
            # print('每次迭代耗时', eplased_times)
            # self.sampler.set_weight(self.trainer.get_state_dict())
            ret = self.predict_client.update_weight(self.trainer.get_state_dict())

            self.logger.info(f"step: {train_step} update weight over ")

    @timer_decorator
    def generate_data(self):
        fragements = self.sampler.get_batch(self.batch_size)
        buffer = Memory()
        import timeit
        for frg in fragements:
            buffer.fstore(frg)
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
    controller.train_run(network_cfg)
    # controller.eval(network_cfg)



