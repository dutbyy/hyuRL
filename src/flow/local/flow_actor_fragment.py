import asyncio
import gym
import torch
import numpy as np
import multiprocessing
import time
import os
import pickle
import logging
from copy import deepcopy
from hyuRL.local.predictor import PredictorClient
from hyuRL.src.tools.common import construct
from hyuRL.src.memory.buffer import Fragment
from hyuRL.src.tools.common import timer_decorator

from typing import Dict, Any, Tuple, Union
from hyuRL.src.flow.drill_plugin.api.flow_api import EnvironmentDescriptor



def sample_target(sample_config):
    async def sample_main(sample_config):
        single_sampler = construct(sample_config)
        await single_sampler.run()
    asyncio.run(sample_main(sample_config))

class SingleActor:
    def __init__(self, total_rewards, datas, data_size, sampling_flag, env_desc, flow_config):
        # print(f"子进程 {os.getpid()} 创建采样器", flush=True)
        self.total_rewards = total_rewards
        self.sampling_flag = sampling_flag
        self.datas = datas
        self.data_size = data_size
        
        self.predictor: PredictorClient = PredictorClient("localhost", 50051)
        self.fragment_size = 32
        self.flow_config = flow_config
        self.env_desc = env_desc

    async def run(self):
        await asyncio.gather(*[self.start_one_task(idx) for idx in range(1)])

    async def start_one_task(self, idx):
        self.env_desc.environment_id_on_this_task += idx
        await self.sample()

    async def sample(self):
        flow_env = self.flow_config['algorithm']['flow_env'](self.env_desc)
        fragment = Fragment()
        while True:
            while self.sampling_flag.value == 0:
                await asyncio.sleep(1)
            total_reward = 0
            flow_env.reset()
            state_dict = flow_env.observe()
            fragment = Fragment()
            while True:
                # while self.sampling_flag.value == 0:
                #     await asyncio.sleep(1)
                episode_done = False if state_dict else True
                logp = 0 
                if not state_dict:
                    piece = (None, None, 0, None, None, 1.0, 0.0, None)
                else:
                    outputs, err = await self.predictor.predict(state_dict['cpdemo']['obs'])
                    action_dict = {"cpdemo": outputs}
                    decoder_mask_dict = {
                        agent_name : flow_env.step(agent_name, agent_command_dict) 
                        for agent_name, agent_command_dict in action_dict.items() 
                    }
                    nstate_dict: Dict[str, Dict[str, Any]] = flow_env.observe()
                    piece = (state_dict['cpdemo']['obs'], action_dict['cpdemo']["action"], 
                        state_dict['cpdemo']['obs']['reward'], logp, decoder_mask_dict['cpdemo']['decoder_mask'],
                        state_dict['cpdemo']['obs']['done'], action_dict['cpdemo']["value"], action_dict['cpdemo']["logits"])

                total_reward += piece[2]
                fragment.store(*piece)
                if fragment.size() >= self.fragment_size or episode_done:
                    with self.data_size.get_lock():
                        self.data_size.value += fragment.size()
                    self.datas.append(pickle.dumps(fragment))
                    fragment = Fragment()
                    if episode_done:
                        self.total_rewards.append(total_reward)
                        break
                    else:
                        fragment.store(*piece)
                state_dict = nstate_dict

class Actor:
    def __init__(self, env_num: int, flow_config):
        self.env_num = env_num
        self.flow_config = flow_config
        '''buffer'''
        self.manager = multiprocessing.Manager()
        self.datas = self.manager.list()
        self.total_rewards = self.manager.list()
        self.data_size = multiprocessing.Value("i", 0)
        self.sampling_flag = multiprocessing.Value("i", 0)
        self.logger = logging.getLogger("Actor")

    @timer_decorator
    def get_batch(self, batch_size=512):
        self.datas[:] = []
        self.data_size.value = 0
        self.sampling_flag.value = 1
        while True:
            print(f"the sum of frament is {self.data_size.value:4}", end="\r", flush=True)
            if self.data_size.value >= batch_size:
                break
            time.sleep(0.02)
        if len(self.total_rewards):
            self.logger.info(
                f"average episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))}"
            )
            print(f"average episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))}")
        self.sampling_flag.value = 0
        self.total_rewards[:] = []
        rets = [pickle.loads(item) for item in self.datas]
        return rets

    def start_sampling(self):
        print("Actor start sampling...")
        self.sampling_flag.value = 1
        ctx = multiprocessing.get_context("spawn")
        sample_args = {
            "class": SingleActor,
            "params": {
                "total_rewards": self.total_rewards, 
                "datas": self.datas,
                "data_size": self.data_size, 
                "sampling_flag": self.sampling_flag, 
                "flow_config": self.flow_config, 
            },
        }
        for idx in range(self.env_num):
            creator = {
                "builder": self.flow_config['builder'],
                "extra_info": {}
            }
            env_desc = EnvironmentDescriptor(1, self.env_num, self.env_num, idx * 5 , idx * 5, idx * 5, creator)
            sample_args['params']['env_desc'] = env_desc
            p = ctx.Process(target=sample_target, args=(sample_args,))
            p.daemon = True
            p.start()
        print("All actor Start Finished!")
