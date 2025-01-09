from __future__ import annotations

import time
import pickle
import logging
import asyncio
import multiprocessing
from collections import defaultdict
from typing import Dict, Any

from hyuRL.src.flow.drill_plugin.api.flow_api import EnvironmentDescriptor
from hyuRL.src.tools.common import construct, timer_decorator, fix_print
from hyuRL.local.flow.flow_predictor import PredictorClient
from hyuRL.src.tools.common import Summary

mean = lambda x : sum(x)/len(x)

def sample_target(sample_config):
    fix_print()
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
        self.fragment_size = 257
        self.flow_config = flow_config
        self.env_desc = env_desc

    async def run(self):
        await asyncio.gather(*[self.start_one_task(idx) for idx in range(4)])

    async def start_one_task(self, idx):
        self.env_desc.environment_id_on_this_task += idx
        await self.sample()

    async def sample(self):
        flow_env = self.flow_config['algorithm']['flow_env'](self.env_desc)
        while True:
            while self.sampling_flag.value == 0:
                await asyncio.sleep(1)
            flow_env.reset()
            state_dict = flow_env.observe()
            fragments = defaultdict(lambda: [])
            while True:
                while self.sampling_flag.value == 0:
                    await asyncio.sleep(1)
                episode_done = False if state_dict else True
                if episode_done:
                    break
                if flow_env.reseted != None:
                    self.total_rewards.append(flow_env.reseted)
                    flow_env.reseted = None
                action_dict = {}
                for agent_name, state in state_dict.items():
                    outputs, err = await self.predictor.predict(state)
                    action_dict[agent_name] = outputs

                decoder_mask_dict = {
                    agent_name : flow_env.step(agent_name, agent_command_dict)
                    for agent_name, agent_command_dict in action_dict.items()
                }
                nstate_dict: Dict[str, Dict[str, Any]] = flow_env.observe()
                piece = [state_dict, action_dict, decoder_mask_dict]
                episode_done = False if nstate_dict else True

                for agent_name in state_dict.keys():
                    agent_piece = [piece[0][agent_name]['obs'], piece[1][agent_name], piece[2][agent_name]]
                    fragments[agent_name].append(agent_piece)

                for agent_name in state_dict.keys():
                    agent_fragments = fragments[agent_name]
                    if len(agent_fragments) >= self.fragment_size or episode_done:
                        with self.data_size.get_lock():
                            self.data_size.value += len(agent_fragments) - 1
                        flow_env.enhance_fragment(agent_name, agent_fragments)
                        self.datas.extend([pickle.dumps(item) for item in agent_fragments ])
                        fragments[agent_name].clear()

                        if not episode_done:
                            agent_piece = [piece[0][agent_name]['obs'], piece[1][agent_name], piece[2][agent_name]]
                            fragments[agent_name].append(agent_piece)

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
                print('')
                break
            time.sleep(0.1)
        if len(self.total_rewards):
            if len(self.total_rewards) > 100:
                self.total_rewards[:] = self.total_rewards[-100:]
            self.logger.info(f"average episode reward is {mean(self.total_rewards):.1f}")
            print(f"average episode reward is {mean(self.total_rewards):.1f}")
            Summary.add_scaler('episode_reward', mean(self.total_rewards))
        rets = [pickle.loads(item) for item in self.datas]
        self.sampling_flag.value = 0
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
