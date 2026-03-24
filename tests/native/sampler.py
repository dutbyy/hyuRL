import asyncio
import gymnasium as gym
import torch
import multiprocessing
import time
import os
import pickle
import logging
from tests.native.predictor import PredictorClient
from hyurl.tools.common import construct
from hyurl.memory.buffer import Fragment

from hyurl.tools.common import timer_decorator



def sample_target(sample_config):
    async def sample_main(sample_config):
        single_sampler = construct(sample_config)
        await single_sampler.run()
    asyncio.run(sample_main(sample_config))


class SingleSampler:
    def __init__(self, total_rewards, datas, data_size, sampling_flag):
        print(f"子进程 {os.getpid()} 创建采样器", flush=True)
        self.total_rewards = total_rewards
        self.sampling_flag = sampling_flag
        self.datas = datas
        self.data_size = data_size
        
        self.predictor: PredictorClient = PredictorClient("localhost", 50051)
        self.fragment_size = 32
    async def run(self):
        await asyncio.gather(*[self.start_one_task() for _ in range(4)])

    async def start_one_task(self):
        await self.sample()

    async def sample(self):
        env = gym.make("CartPole-v1")
        fragment = Fragment()
        while True:
            while self.sampling_flag.value == 0:
                await asyncio.sleep(1)
            total_reward = 0
            state, _ = env.reset()
            while True:
                if self.sampling_flag.value == 0:
                    fragment = Fragment()
                    await asyncio.sleep(1)
                    continue
                
  
                outputs, err = await self.predictor.predict({"common": state})
                nstate, reward, done, truncted, info = env.step(outputs["action"]["meta_action"])

            
                total_reward += 1
                done = done or truncted
                action_mask = {"meta_action": torch.tensor(1)}
                logp = 0 
                fragment.store(
                    {"common": state},
                    outputs["action"],
                    reward,
                    logp,
                    action_mask,
                    1 if done else 0,
                    outputs["value"],
                    outputs["logits"],
                )
                if fragment.size() >= self.fragment_size or done:

                    fsize = fragment.size()
                    with self.data_size.get_lock():
                        self.data_size.value += fsize - 1
                    # fragment = gzip.compress(pickle.dumps(fragment))
                    fragment = pickle.dumps(fragment)
                    self.datas.append(fragment)
                    fragment = Fragment()
                    if not done:
                        fragment.store(
                            {"common": state},
                            outputs["action"],
                            reward,
                            logp,
                            action_mask,
                            1 if done else 0,
                            outputs["value"],
                            outputs["logits"],
                        )
                    if done:
                        self.total_rewards.append(total_reward)
                        break
                state = nstate


class Sampler:
    def __init__(self, delayed_policy=None, env_config=None, num_processes=4):
        self.manager = multiprocessing.Manager()
        self.model_dict = self.manager.dict()
        self.datas = self.manager.list()
        self.total_rewards = self.manager.list()
        self.data_size = multiprocessing.Value("i", 0)
        self.sampling_flag = multiprocessing.Value("i", 1)
        self.num_processes = num_processes
        self.logger = logging.getLogger(f"model_learn")

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
                f"average episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))} \n"
            )
        self.sampling_flag.value = 0
        self.total_rewards[:] = []
        rets = [pickle.loads(item) for item in self.datas]
        return rets


    def start_sampling(self):
        print(f"Main Process {os.getpid()} start sampling...")
        self.sampling_flag.value = 1
        ctx = multiprocessing.get_context("spawn")
        sample_args = {
            "class": SingleSampler,
            "params": {
                "total_rewards": self.total_rewards,
                "datas": self.datas,
                "data_size": self.data_size,
                "sampling_flag": self.sampling_flag,
            },
        }
        for _ in range(self.num_processes):
            p = ctx.Process(target=sample_target, args=(sample_args,))
            p.daemon = True
            p.start()
        print("All SingleSampler Start Finished!")

