from __future__ import annotations

from typing import Dict
from copy import deepcopy
from matplotlib.testing.jpl_units import EpochConverter
import numpy as np
import random
from hyuRL.local.flow.flow_actor import Actor
from hyuRL.local.flow.flow_learner import LocalLearner
from hyuRL.local.flow.flow_predictor import PredictorClient
from hyuRL.src.tools.common import fix_print


def datas_prefix(datas, batch_size=1024):
    assert batch_size <= len(datas)
    datas = random.sample(datas, batch_size)

    def recursion(demo, origin):
        if isinstance(demo, dict):
            return {k: recursion(v, [it[k] for it in origin]) for k, v in demo.items()}
        elif isinstance(demo, list):
            return [recursion(item, [it[i] for it in origin]) for i, item in enumerate(demo)]
        elif isinstance(demo, np.ndarray):
            return np.stack(origin, 0)
        else:
            raise Exception(f"unsupport type: {type(demo)}")

    return recursion(deepcopy(datas[0]), datas)


def trans2tensor(nested_structure):
    import tree
    import torch

    return tree.map_structure(lambda x: torch.from_numpy(x).cuda(), nested_structure)


class LocalMaster:
    def __init__(self, flow_config: Dict, epoch_num=10, sample_size=128, env_num=4, batch_size=64, save_interval=10):
        self.predictor = PredictorClient("localhost", 50051, False)
        self.actor = Actor(env_num=4, flow_config=flow_config)
        self.learner = LocalLearner(flow_config)
        self.env_num = env_num
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.save_interval = save_interval
        self.train_step = 0

    def run(self):
        self.actor.start_sampling()
        for model_name in self.learner.model_names:
            weights = self.learner.get_weights(model_name)
            self.predictor.update_weight(model_name, weights)
            self.learner.flow_model_dic[model_name].save_weights()

        def wrapper(datas, mini_batch):
            t = mini_batch
            idx = 0
            while t <= len(datas):
                yield idx, datas_prefix(datas[t - mini_batch : t], mini_batch)
                t += mini_batch
                idx += 1

        while True:
            self.train_step += 1
            datas = self.actor.get_batch(self.sample_size)
            for epoch in range(10):
                random.shuffle(datas)
                for idx, train_datas in wrapper(datas, self.batch_size):
                    self.learner.train(self.learner.model_names[0], train_datas)
                    print(f"training step: {self.train_step} epoch: {epoch+1} times: {idx}", end="\r", flush=True)
            print()
            for model_name in self.learner.model_names:
                weights = self.learner.get_weights(model_name)
                self.predictor.update_weight(model_name, weights, f"learn_step: {self.train_step}")
                if self.save_interval and self.train_step % self.save_interval == 0:
                    self.learner.flow_model_dic[model_name].save_weights()


def main(flow_config):
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    master = LocalMaster(flow_config, sample_size=4096, batch_size=1024)
    master.run()


if __name__ == "__main__":
    from hyuRL.local.flow.env_config import flow_config
    from hyuRL.src.tools.common import Summary
    Summary.setpath('beamrider-v4-sb3-env')
    fix_print()
    main(flow_config)
