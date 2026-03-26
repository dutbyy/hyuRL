from __future__ import annotations

from typing import Dict
from copy import deepcopy
import numpy as np
import random
from tests.local.sampler import LocalSampler
from tests.local.learner import LocalLearner
from hyurl.tools.common import fix_print, Summary


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


class LocalTrainer:
    def __init__(self, flow_config: Dict, epoch_num=10, sample_size=128, batch_size=64, save_interval=10):
        self.sampler = LocalSampler(flow_config)
        self.learner = LocalLearner(flow_config)
        self.flow_config = flow_config
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.save_interval = save_interval
        self.train_step = 0

    def run(self):
        self.sampler.reset()
        
        def wrapper(datas, mini_batch):
            t = mini_batch
            idx = 0
            while t <= len(datas):
                yield idx, datas_prefix(datas[t - mini_batch : t], mini_batch)
                t += mini_batch
                idx += 1

        while True:
            self.train_step += 1
            datas = self.sampler.get_batch(self.sample_size)
            for epoch in range(self.epoch_num):
                random.shuffle(datas)
                for idx, train_datas in wrapper(datas, self.batch_size):
                    for model_name in self.learner.model_names:
                        self.learner.train(model_name, train_datas)
            for model_name in self.learner.model_names:
                weights = self.learner.get_weights(model_name)
                if self.save_interval and self.train_step % self.save_interval == 0:
                    self.learner.flow_model_dic[model_name].save_weights()


def main(flow_config):
    trainer = LocalTrainer(flow_config, epoch_num=1, sample_size=4096, batch_size=4096)
    trainer.run()


if __name__ == "__main__":
    from tests.local.entry import flow_config
    Summary.setpath('cartpole-v1-env')
    fix_print()
    main(flow_config)
