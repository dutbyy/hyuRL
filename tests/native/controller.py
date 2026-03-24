import logging
import numpy as np 
np.bool8 = np.bool_
logger = logging.getLogger("native")
logger.setLevel(logging.INFO)


from tests.native.sampler import Sampler
from tests.native.trainer import LocalTrainer
from hyurl.algo.PPOPolicy import PPOPolicy
from hyurl.memory.buffer import Memory
from hyurl.tools.common import timer_decorator

from tests.native.predictor import PredictorClient


def trans2tensor(nested_structure):
    import tree
    import torch
    return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)


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
                "[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
            )
            handler.setFormatter(formatter)  
            self.logger.addHandler(handler)
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Logger Init Finished")

    def eval(self, policy_config):
        delayed_policy = {
            "class": PPOPolicy,
            "params": {"network_config": policy_config, "device": "cpu"},
        }
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=1)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()

    # @timer_decorator
    def train_run(self, policy_config):
        delayed_policy = {
            "class": PPOPolicy,
            "params": {"network_config": policy_config, "device": "cpu"},
        }
        self.sampler = Sampler(delayed_policy=delayed_policy, num_processes=8)

        delayed_policy = {
            "class": PPOPolicy,
            "params": {
                "network_config": policy_config,
                "device": "cpu",
                "trainning": True,
            },
        }
        self.trainer = LocalTrainer(delayed_policy=delayed_policy)
        self.predict_client = PredictorClient("localhost", 50051, False)

        self.sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            train_data = self.generate_data()
            # 定义一个递归函数来处理嵌套结构

            train_data = trans2tensor(train_data)
            eplased_times = self.trainer.train(train_data)
            # print('每次迭代耗时', eplased_times)
            # self.sampler.set_weight(self.trainer.get_state_dict())
            weights = self.trainer.get_state_dict()
            ret = self.predict_client.update_weight("", weights)

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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    from tests.native.net import network_cfg
    controller = Controller()
    controller.train_run(network_cfg)
