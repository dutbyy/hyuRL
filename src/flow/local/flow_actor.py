import asyncio
import multiprocessing
import time
import os
import pickle
import logging
from local.predictor import PredictorClient
from copy import deepcopy
from src.tools.common import timer_decorator
from src.tools.common import construct
from typing import Dict, Any, Tuple, Union
from hyuRL.src.flow.drill_plugin.api.flow_api import EnvironmentDescriptor
from hyuRL.src.flow.local.buffer import Fragment

def sample_target(sample_config):
    async def sample_main(sample_config):
        single_sampler = construct(sample_config)
        await single_sampler.run()
    asyncio.run(sample_main(sample_config))

class SingleActor:
    def __init__(self, env_desc,  total_rewards, datas, data_size, sampling_flag, flow_config):
        print(f"子进程 {os.getpid()} 创建采样器", flush=True)
        self.env_desc = env_desc
        self.total_rewards = total_rewards
        self.sampling_flag = sampling_flag
        self.datas = datas
        self.data_size = data_size
        self.flow_env = flow_config['algorithm']['flow_env'](env_desc)
        self.predictor: PredictorClient = PredictorClient("localhost", 50051)
        self.need_update_weight = False
        self.fragment_size = 16

    async def run(self):
        # await asyncio.gather(*[self.start_one_task() for _ in range(1)])
        await asyncio.gather(*[self.start_one_task() for _ in range(10)])
        # await self.sample()

    async def start_one_task(self):
        await self.sample()

    async def sample(self):
        fragment_list = []
        # fragment_list = Fragment()
        while True:
            # print("sampling 1  ")
            # if not self.sampling_flag.value :
                # time.sleep(0.1)
                # continue
            # print("sampling 2")
            self.flow_env.reset()
            while True:
                state_dict: Dict[str, Dict[str, Any]] = self.flow_env.observe()
                if not state_dict:
                    fragment = [ {"reward": 0.0, "value": 0.0, "done": 1.0},
                                action_dict['cpdemo'], 
                                decoder_mask_dict['cpdemo']
                            ]
                    fragment_list.append(fragment)
                    if fragment_list:
                        self.flow_env.enhance_fragment("cpdemo", fragment_list)
                        self.datas.extend([pickle.dumps(it) for it in fragment_list])
                        fragment_list.clear()
                    break
                action_dict, err = await self.predictor.predict(state_dict['cpdemo']['obs'])
                action_dict = {"cpdemo": action_dict}
                decoder_mask_dict = {}
                # print("sampling 3")
                for agent_name, agent_command_dict in action_dict.items():
                    agent_decoder_mask = self.flow_env.step(agent_name, agent_command_dict)
                    decoder_mask_dict[agent_name] = agent_decoder_mask
                fragment = [ state_dict['cpdemo']['obs'], action_dict['cpdemo'], decoder_mask_dict['cpdemo']]
                # print(state_dict['cpdemo']['obs']['done'].item())
                fragment_list.append(fragment)
                if len(fragment_list) > self.fragment_size:
                    self.flow_env.enhance_fragment("cpdemo", fragment_list)
                    self.datas.extend([pickle.dumps(it) for it in fragment_list])
                    fragment_list.clear()

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
            print(f"the sum of frament is {len(self.datas)}", end="\r", flush=True)
            if len(self.datas) >= batch_size:
                break
            time.sleep(0.1)
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
        self.sampling_flag.value = 0
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
            env_desc = EnvironmentDescriptor(1, self.env_num, self.env_num, idx, idx, idx, creator)
            sample_args['params']['env_desc'] = env_desc
            p = ctx.Process(target=sample_target, args=(sample_args,))
            p.daemon = True
            p.start()
        print("All actor Start Finished!")
