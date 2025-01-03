from typing import Dict
from hyuRL.src.flow.local.flow_actor import Actor
from hyuRL.src.flow.local.flow_predictor import PredictorClient
from hyuRL.src.flow.local.flow_learner import LocalLearner
from copy import deepcopy
import numpy as np
import random
from hyuRL.src.memory.buffer import Memory

def datas_prefix(datas, batch_size=1024):
    assert batch_size <= len(datas)
    datas = random.sample(datas, batch_size)
    def dg(demo, origin):
        if isinstance(demo, dict):
            for k, v in demo.items():
                # print(f"cacl dg {k}")
                # print( [it[k] for it in origin])
                # print( [it[k].shape for it in origin])
                demo[k] = dg(v, [it[k] for it in origin])
        elif isinstance(demo, list):
            for idx, item in enumerate(demo):
                demo[idx] = dg(item, [it[idx] for it in origin])
        elif isinstance(demo, np.ndarray):
            return np.stack(origin, 0)
        else:
            raise Exception(f"unsupport type: {type(demo)}")
        return demo
    demo = deepcopy(datas[0])
    return dg(demo, datas)

def trans2tensor(nested_structure):
    import tree
    import torch
    return tree.map_structure(lambda x: torch.from_numpy(x).cuda(), nested_structure)

class LocalMaster:
    def __init__(self, flow_config: Dict):
        self.predictor = PredictorClient("localhost", 50051, False)
        self.actor = Actor(env_num=1, flow_config=flow_config)
        self.learner = LocalLearner(flow_config)
        self.batch_size = 1024
        self.train_step = 0

    def run(self):
        self.actor.start_sampling()
        for model_name in self.learner.model_names:
            weights = self.learner.get_weights(model_name)
            self.predictor.update_weight(model_name, weights)
            self.learner.flow_model_dic[model_name].save_weights()

        def wrapper(datas, mini_batch):
            t = mini_batch
            while t <= len(datas):
                yield datas_prefix(datas[t-mini_batch: t], mini_batch)
                t += mini_batch

        while True:
            self.train_step += 1
            datas = self.actor.get_batch(self.batch_size)
            for epoch in range(4):
                random.shuffle(datas)
                idx = 0
                for train_datas in wrapper(datas, 128):
                    idx+=1
                    print(f"training epoch: {epoch+1} times: {idx}", end='\r', flush=True)
                    ret = self.learner.train(self.learner.model_names[0], train_datas)
            print()
            print(f"train step: {self.train_step}")
            for model_name in self.learner.model_names:
                weights = self.learner.get_weights(model_name)
                self.predictor.update_weight(model_name, weights)
                if self.train_step % 2 == 0:
                    self.learner.flow_model_dic[model_name].save_weights()

def main(flow_config):
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    master = LocalMaster(flow_config)
    master.run()

def fix_print():
    import builtins, os
    origin_print = builtins.print
    def custom_print(*args, **kwargs):
        import datetime
        import inspect
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        prefix = f"[{timestamp}] [{os.path.basename(caller.filename)}:{caller.lineno}]"
        origin_print(prefix, *args, **kwargs)
    builtins.print = custom_print


if __name__ == '__main__':
    import torch
    import numpy as np
    import random

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    fix_print()
    from hyuRL.src.flow.local.env_config import flow_config
    main(flow_config)