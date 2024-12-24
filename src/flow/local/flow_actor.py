import asyncio
import torch
import multiprocessing
import time
import os
import timeit
import pickle
import logging
from local.predictor import PredictorClient
from src.tools.gpu import auto_move
from src.memory.buffer import Fragment
from copy import deepcopy
from src.tools.common import timer_decorator
from src.tools.common import construct
from typing import Dict, Any, Tuple, Union


def sample_target(sample_config):
    async def sample_main(sample_config):
        single_sampler = construct(sample_config)
        await single_sampler.run()
    asyncio.run(sample_main(sample_config))

class SingleActor:
    def __init__(self, env_id,  total_rewards, datas, data_size, sampling_flag, flow_config):
        print(f"子进程 {os.getpid()} 创建采样器", flush=True)
        self.env_id = env_id
        self.total_rewards = total_rewards
        self.sampling_flag = sampling_flag
        self.datas = datas
        self.data_size = data_size
        self.flow_env = construct(flow_config['algorithm']['flow_env'])
        self.predictor: PredictorClient = PredictorClient("localhost", 50051)
        self.need_update_weight = False
        self.fragment_size = 32

    async def run(self):
        await asyncio.gather(*[self.start_one_task() for _ in range(4)])

    async def start_one_task(self):
        await self.sample()

    async def sample(self):
        fragment_list = []
        self.flow_env.reset()
        while True:
            state_dict: Dict[str, Dict[str, Any]] = self.flow_env.observe()
            action_dict = self.predictor.predict(state_dict)
            decoder_mask_dict = {}
            for agent_name, agent_command_dict in action_dict.items():
                agent_decoder_mask = self.flow_env.step(agent_name, agent_command_dict)
                decoder_mask_dict[agent_name] = agent_decoder_mask
            fragment = [ state_dict, action_dict, decoder_mask_dict]
            fragment_list.append(fragment)
            if len(fragment_list) > self.fragment_size:
                self.flow_env.enhance_fragment(fragment_list)
                self.datas.extend(fragment_list)
                with self.data_size.get_lock():
                        self.data_size.value += len(fragment_list) - 1

class Actor:
    def __init__(self, env_num: int, extra_info: Dict):
        self.env_num = env_num
        '''buffer'''
        self.manager = multiprocessing.Manager()
        self.datas = self.manager.list()
        self.total_rewards = self.manager.list()
        self.data_size = multiprocessing.Value("i", 0)
        self.sampling_flag = multiprocessing.Value("i", 1)
        self.logger = logging.getLogger("Actor")

    @timer_decorator
    def get_batch(self, batch_size=512):
        self.datas[:] = []
        self.data_size.value = 0
        self.sampling_flag.value = 1
        while True:
            print(f"the sum of frament is {self.data_size.value}", end="\r", flush=True)
            if self.data_size.value >= batch_size:
                break
            time.sleep(0.01)
        if len(self.total_rewards):
            self.logger.info(
                f"average episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))}"
            )
        self.sampling_flag.value = 0
        self.total_rewards[:] = []
        rets = [pickle.loads(item) for item in self.datas]
        return rets

    def start(self):
        print("Actor start sampling...")
        self.sampling_flag.value = 1
        ctx = multiprocessing.get_context("spawn")
        sample_args = {
            "class": SingleActor,
            "params": {
                "env_id": -1,
                "delayed_policy": self.delayed_policy,
                "total_rewards": self.total_rewards,
                "datas": self.datas,
                "data_size": self.data_size,
                "sampling_flag": self.sampling_flag,
                "weights_dict": self.model_dict,
            },
        }
        for idx in range(self.env_num):
            sample_args['params']['env_id'] = idx
            p = ctx.Process(target=sample_target, args=(sample_args,))
            p.daemon = True
            p.start()
        print("All actor Start Finished!")
