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
    def __init__(self, flow_config: Dict, model_name: str, builder):
        self.predictor = PredictorClient("localhost", 50051, False)
        self.actor = Actor(env_num=8, flow_config=flow_config)
        self.learner = LocalLearner(flow_config, model_name, builder)
        self.batch_size = 4096
        self.train_step = 0
        
    def run(self):
        self.actor.start_sampling()
        weights = self.learner.get_weights()
        self.predictor.update_weight("cp_model", weights)

        while True:
            self.train_step += 1
            datas = self.actor.get_batch(self.batch_size)
            train_datas = datas_prefix(datas)
            self.learner.train(train_datas)
            print(f"train step :{self.train_step}")
            weights = self.learner.get_weights()
            self.predictor.update_weight("cp_model", weights)


def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    from hyuRL.example.cartpole.entry import flow_config, builder
    master = LocalMaster(flow_config, 'cp_model', builder)
    master.run()
    
if __name__ == '__main__':
    main()