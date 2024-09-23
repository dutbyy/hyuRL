import asyncio
import gym
import torch
import numpy as np
import multiprocessing
import time
import os
import timeit
import pickle
import logging
from local.predictor import PredictorClient
from src.tools.gpu import auto_move
from src.tools.common import construct
from src.memory.buffer import Fragment
from src.algo import PPOPolicy
from copy import deepcopy
from src.tools.common import timer_decorator


def sample_target(sample_config):
    async def sample_main(sample_config):
        single_sampler = construct(sample_config)
        await single_sampler.run()

    asyncio.run(sample_main(sample_config))


class SingleSampler:
    def __init__(
        self,
        delayed_policy,
        total_rewards,
        datas,
        data_size,
        sampling_flag,
        weights_dict,
    ):
        print(f"子进程 {os.getpid()} 创建采样器", flush=True)
        self.total_rewards = total_rewards
        self.sampling_flag = sampling_flag
        self.datas = datas
        self.data_size = data_size

        self.predictor: PredictorClient = PredictorClient("localhost", 50051)
        self.policy = construct(delayed_policy)
        self.need_update_weight = False
        self.weights_dict = weights_dict

    async def run(self):
        await asyncio.gather(*[self.start_one_task() for _ in range(4)])

    async def start_one_task(self):
        await self.sample()

    async def sample(self):
        env = gym.make("CartPole-v1")
        infer_eplased_times = []
        env_eplased_times = []
        fragment = Fragment()
        while True:
            while self.sampling_flag.value == 0:
                self.need_update_weight = True
                await asyncio.sleep(1)
            if self.need_update_weight:
                self.need_update_weight = False
                if "weight" in self.weights_dict:
                    state_dict = deepcopy(self.weights_dict["weight"])
                    self.policy._network.load_state_dict(state_dict)
            total_reward = 0
            state, _ = env.reset()
            while True:
                if self.sampling_flag.value == 0:
                    self.need_update_weight = True
                    fragment = Fragment()
                    await asyncio.sleep(1)
                    continue

                outputs, eplased_time = await self.predictor.predict(
                    {"feature_a": torch.Tensor(state)}
                )
                infer_eplased_times.append(eplased_time)

                _1 = timeit.default_timer() * 1000
                # print(outputs)
                nstate, reward, done, truncted, info = env.step(
                    outputs["action"]["action"].item()
                )

                _2 = timeit.default_timer() * 1000
                env_eplased_times.append(_2 - _1)

                total_reward += 1
                done = done or truncted
                action_mask = {"action": torch.tensor(1)}
                logp = 0  # self.policy._network.log_probs(outputs['logits'], outputs['action'], action_mask)
                # logp =  await self.predictor.log_probs(outputs['logits'], outputs['action'], action_mask)
                fragment.store(
                    {"feature_a": torch.Tensor(state)},
                    outputs["action"],
                    reward,
                    logp,
                    action_mask,
                    1 if done else 0,
                    outputs["value"],
                    outputs["logits"],
                )
                if fragment.size() >= 16 or done:

                    fsize = fragment.size()
                    with self.data_size.get_lock():
                        self.data_size.value += fsize - 1
                    # fragment = gzip.compress(pickle.dumps(fragment))
                    fragment = pickle.dumps(fragment)
                    self.datas.append(fragment)
                    fragment = Fragment()
                    avg_infer_time = round(
                        sum(infer_eplased_times) / len(infer_eplased_times), 1
                    )
                    avg_env_time = round(
                        sum(env_eplased_times) / len(env_eplased_times), 1
                    )
                    # print(f'collect a fragment eplased time: avg infer time: {avg_infer_time} avg env time: {avg_env_time}')
                    infer_eplased_times = []
                    env_eplased_times = []
                    if not done:
                        fragment.store(
                            {"feature_a": torch.Tensor(state)},
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
        self.delayed_policy = delayed_policy
        # self.predictor = predictor.PredictorClient('localhost', 50051)
        # self.model_dict['predictor'] = self.predictor
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
            time.sleep(0.1)
        if len(self.total_rewards):
            self.logger.info(
                f"average episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))}"
            )
        self.sampling_flag.value = 0
        self.total_rewards[:] = []
        rets = [pickle.loads(item) for item in self.datas]
        return rets

    def set_weight(self, weight):
        self.model_dict["weight"] = auto_move(weight, "cpu")

    def start_sampling(self):
        print(f"Main Process {os.getpid()} start sampling...")
        self.sampling_flag.value = 1
        ctx = multiprocessing.get_context("spawn")
        sample_args = {
            "class": SingleSampler,
            "params": {
                "delayed_policy": self.delayed_policy,
                "total_rewards": self.total_rewards,
                "datas": self.datas,
                "data_size": self.data_size,
                "sampling_flag": self.sampling_flag,
                "weights_dict": self.model_dict,
            },
        }
        for _ in range(self.num_processes):
            p = ctx.Process(target=sample_target, args=(sample_args,))
            p.daemon = True
            p.start()
        print("All SingleSampler Start Finished!")


if __name__ == "__main__":

    class EmptyClass:
        pass

    np.set_printoptions(suppress=True)
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    self = EmptyClass()
    from src.network import *
    from src.algo.PPOPolicy import PPOPolicy

    network_cfg = {
        "encoder_demo": {
            "class": CommonEncoder,
            "params": {
                "in_features": 4,
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ["feature_a"],
        },
        "aggregator": {
            "class": DenseAggregator,
            "params": {
                "in_features": 128,
                "hidden_layer_sizes": [256, 128],
                "output_size": 256,
            },
            "inputs": ["encoder_demo"],
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features": 256,
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ["aggregator"],
        },
        "action": {
            "class": CategoricalDecoder,
            "params": {
                "n": 2,
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ["aggregator"],
        },
    }
    delayed_policy = {"class": PPOPolicy, "params": {"policy_config": network_cfg}}
    self.delayed_policy = delayed_policy
    mgr = multiprocessing.Manager()
    self.total_rewards = mgr.list()
    self.datas = mgr.list()
    self.model_dict = mgr.dict()
    self.data_size = multiprocessing.Value("i", 0)
    self.sampling_flag = multiprocessing.Value("i", 1)

    start_config = {
        "class": SingleSampler,
        "params": {
            "delayed_policy": self.delayed_policy,
            "total_rewards": self.total_rewards,
            "datas": self.datas,
            "data_size": self.data_size,
            "sampling_flag": self.sampling_flag,
            "weights_dict": self.model_dict,
        },
    }
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=sample_target, args=(start_config,))
    p.daemon = True
    p.start()
    while True:
        time.sleep(5)
        print(
            f"sample config is {(self.sampling_flag.value)} {(self.data_size.value)}",
            flush=True,
        )
