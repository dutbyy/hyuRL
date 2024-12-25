from hyuRL.src.algo.PPOPolicy import PPOPolicy
from typing import Dict
from hyuRL.src.flow.local.flow_actor import Actor
from hyuRL.src.flow.local.flow_predictor import PredictorClient
from hyuRL.src.flow.local.flow_learner import LocalLearner 
from copy import deepcopy
import numpy as np

def datas_prefix(datas):
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
    # batch_size = len(datas)
    demo = deepcopy(datas[0])
    return dg(demo, datas)


class LocalMaster:
    def __init__(self, flow_config: Dict, model_name: str, builder):
        self.predictor = PredictorClient("localhost", 50051, False)
        self.actor = Actor(env_num=1, flow_config=flow_config)
        self.learner = LocalLearner(flow_config, model_name, builder)
        self.batch_size = 4096
        
    def train(self, train_data):
        return self.flow_model.learn(train_data)

    def run(self):
        self.actor.start()
        while True:
            datas = self.actor.get_batch(self.batch_size)
            datas = datas_prefix(datas)
            self.learner.train(datas)
            ret = self.predictor.update_weight(self.learner.get_weights())


def main():
    from hyuRL.example.cartpole.entry import flow_config, builder
    master = LocalMaster(flow_config, 'cp_model', builder)
    master.run()
    
if __name__ == '__main__':
    main()