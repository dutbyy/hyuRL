from copy import deepcopy
import gym
from lib.memory.buffer import Fragment
import torch
import numpy as np
import multiprocessing
import time
import os
import timeit
import pickle
from lib.tools.gpu import auto_move
from lib.tools.common import construct
import gzip

def sample_target(sample_config):
    single_sampler = construct(sample_config)
    single_sampler.run()

class SingleSampler:
    def __init__(self, delayed_policy, total_rewards, datas, data_size, sampling_flag, weights_dict):
        print(f"子进程 {os.getpid()} 创建采样器")
        self.total_rewards  = total_rewards
        self.sampling_flag  = sampling_flag
        self.weights_dict   = weights_dict
        self.datas          = datas
        self.data_size      = data_size
        self.policy:PPOPolicy = construct(delayed_policy)
        
        self.need_update_weight  = False

    def run(self):
        # print("开始采样")
        fragment_generator = self.sample()
        while True:
            # print(f"采样标志位 sample flag value : {self.sampling_flag.value}")
            try:
                # print(f"采样标志位 sample flag value : {self.sampling_flag.value}")

                if self.sampling_flag.value:
                    if self.need_update_weight:
                        self.need_update_weight = False
                        if 'weight' in self.weights_dict:    
                            state_dict = deepcopy(self.weights_dict['weight'])
                            self.policy._network.load_state_dict(state_dict)
                            self.need_update_weight = False
                    next(fragment_generator)
                else:
                    self.need_update_weight = True
                    time.sleep(1)  # 如果不进行采样，那么就让进程睡眠
            except Exception as e:
                print(f"worker Error : {e}")
                raise e

    def sample(self):
        self.env = gym.make('CartPole-v1')
        self.need_update_weight = False
        eplased_times = []
        env_eplased_times = [] 
        tbegin = timeit.default_timer() * 1000
        fragment = Fragment()
        while True:
            while self.sampling_flag.value == 0:
                time.sleep(1)
                
            if self.need_update_weight:
                self.need_update_weight = False
                if 'weight' in self.weights_dict:    
                    state_dict = deepcopy(self.weights_dict['weight'])
                    self.policy._network.load_state_dict(state_dict)
                    self.need_update_weight = False
                    
            total_reward = 0
            _1 = timeit.default_timer() * 1000
            state, _ = self.env.reset()
            _2 = timeit.default_timer() * 1000
            env_eplased_times.append(_2-_1)
            # print("reset an environment")
            while True: 
                if self.sampling_flag.value == 0:
                    self.need_update_weight = True
                    fragment = Fragment()
                    eplased_times = []
                    env_eplased_times = []
                    tbegin = timeit.default_timer() *1000
                    break                    
                    
                # print('env a step')
                outputs, eplased_time = self.policy.inference({"feature_a": torch.Tensor(state) * torch.ones(1024, 1)})
                eplased_times.append(eplased_time)
                # print("inference ok")
                _1 = timeit.default_timer() * 1000
                nstate, reward, done, truncted, info = self.env.step(outputs['action']['action'][0].item())
                _2 = timeit.default_timer() * 1000
                env_eplased_times.append(_2-_1)
                reward *= 1 
                # reward -= nstate[0]  * 0.1
                # reward -= nstate[1]  * 0.1
                # reward -= nstate[2]  * 0.3
                # reward -= nstate[3]  * 0.3
                total_reward += 1
                done = done or truncted
                action_mask = {"action": torch.tensor(1)}
                ta = timeit.default_timer() * 1000
                logp = self.policy._network.log_probs( outputs['logits'], outputs['action'], action_mask)
                tb = timeit.default_timer() * 1000
                eplased_time = tb-ta
                eplased_times[-1] += eplased_time
                fragment.store( {"feature_a": torch.Tensor(state)}, outputs['action'], reward, logp, action_mask, 
                               1 if done else 0 , outputs['value'], outputs['logits'])
                if fragment.size() >= 16:
                    
                    fsize = fragment.size()

                    with self.data_size.get_lock():
                        self.data_size.value += (fragment.size()-1)
                    from pympler import asizeof
                    fragment = gzip.compress(pickle.dumps(fragment))
                    fsize = asizeof.asizeof(fragment)
                    
                    A = timeit.default_timer() * 1000
                    self.datas.append(fragment)
                    B= timeit.default_timer() * 1000
                    # print(f'追加一个 fragment {fsize} 耗时 {(B-A)}')
                    tend = timeit.default_timer() * 1000

                    # print(f'收集一个一个 fragment {fsize} 耗时 {round(tend - tbegin, 6)} ms')
                    print(f'推理平均耗时为 {round(np.array(eplased_times).mean(), 1)} ms')
                    print(f'推理总耗时为 {round(np.array(eplased_times).sum(), 1)} ms')
                    # print(f'env总耗时为 {round(np.array(env_eplased_times).sum(), 1)} ms')                   
                    # print(f'追加一个 fragment 总耗时{round(B-A, 1)} ms')
                    # print(f'推理耗时占比为 {round(np.array(eplased_times).sum() / round(tend - tbegin, 6) * 100 , 1)}% ')
                    # print(f'env总耗时比 {round(np.array(env_eplased_times).sum() / round(tend - tbegin, 6) * 100 ,1)}% ')
                    fragment = Fragment()
                    eplased_times = []
                    env_eplased_times = []
                    tbegin = timeit.default_timer() *1000
                    if not done:
                        fragment.store( {"feature_a": torch.Tensor(state)}, outputs['action'], reward, logp, action_mask, 
                               1 if done else 0 , outputs['value'], outputs['logits'])
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
        self.data_size = multiprocessing.Value('i', 0)
        self.sampling_flag = multiprocessing.Value('i', 1)
        self.num_processes = num_processes
        self.delayed_policy = delayed_policy
                   
    def get_batch(self, batch_size=512):
        # print("callling get batch")
        a = timeit.default_timer()
        while True:
            # print(f"eplased time : {int((b-a) * 1000) }ms,  {int((c-b) * 1000) }ms")
            print(f"the sum of frament is {self.data_size.value}", end='\r', flush=True)
            if self.data_size.value >= batch_size:
                break
            time.sleep(0.02)
        if len(self.total_rewards):
            print(f'\naverage episode reward is {int(sum(self.total_rewards) /  len(self.total_rewards))}')
        self.sampling_flag.value = 0
        b = timeit.default_timer()
        print(f"收集一个一个batch的数据需要耗时 :{b*1000 - a*1000} ms")
        return [pickle.loads(gzip.decompress(item)) for item in self.datas] 
   
    # @classmethod
    # def sample(cls, policy, total_rewards, data, data_size):
    #     class EmptyClass:
    #         pass

    #     self = EmptyClass()
    #     self.env = gym.make('CartPole-v1')
    #     self.policy = policy
    #     self.total_rewards = total_rewards 
    #     self.data = data 
    #     self.data_size = data_size 
        
        
    #     # self.policy = self.model_dict['policy']
    #     self.need_update_weight = True
    #     print(f'Process {os.getpid()} my policy is ', self.policy)
        
    #     # time.sleep(10)
    #     eplased_times = []
    #     env_eplased_times = []
    #     print(f"Process {os.getpid()} is sampling...")
    #     tbegin = timeit.default_timer()
    #     fragment = Fragment()
    #     while True:
    #         total_reward = 0
    #         state, _ = self.env.reset()
    #         # print("reset an environment")
    #         while True: 
    #             # print('env a step')
    #             outputs, eplased_time = self.policy.inference({"feature_a": torch.Tensor(state)})
    #             eplased_times.append(eplased_time)
    #             # print("inference ok")
    #             _1 = timeit.default_timer() * 1000
    #             nstate, reward, done, truncted, info = self.env.step(outputs['action']['action'].item())
    #             _2 = timeit.default_timer() * 1000
    #             env_eplased_times.append(_2-_1)
    #             reward *= 1
    #             reward -= nstate[0]  * 0.1
    #             reward -= nstate[1]  * 0.1
    #             reward -= nstate[2]  * 0.3
    #             reward -= nstate[3]  * 0.3
    #             total_reward += 1
    #             done = done or truncted
    #             action_mask = {"action": torch.tensor(1)}
    #             logp = self.policy._network.log_probs( outputs['logits'], outputs['action'], action_mask)

    #             fragment.store( {"feature_a": torch.Tensor(state)}, outputs['action'], reward, logp, action_mask, 
    #                            1 if done else 0 , outputs['value'], outputs['logits'])
    #             if fragment.size() >= 128:
    #                 tend = timeit.default_timer()
                    
    #                 # print(f'收集一个一个 fragment {fragment.size()} 耗时 {round(tend - tbegin, 6)*1000}ms')
    #                 # print(f'推理平均耗时为 {np.array(eplased_times).mean()} ms')
    #                 # print(f'推理总耗时为 {np.array(eplased_times).sum()} ms')
    #                 # print(f'env总耗时为 {np.array(env_eplased_times).sum()} ms')
    #                 # print(f'推理耗时占比为 {np.array(eplased_times).sum() / round(tend - tbegin, 6) / 10 }% ')
    #                 # print(f'env总耗时比 {np.array(env_eplased_times).sum() / round(tend - tbegin, 6) / 10 }% ')
                    
    #                 eplased_times = []
    #                 self.data_size.value += fragment.size()
    #                 from pympler import asizeof
    #                 fragment = pickle.dumps(auto_move(fragment,'cpu'))
    #                 fsize = asizeof.asizeof(fragment)
    #                 A = timeit.default_timer()
    #                 self.data.append(fragment)
    #                 B= timeit.default_timer()
    #                 # print(f'追加一个 fragment {fsize} 耗时 {(B-A)}')
    #                 yield
    #                 fragment = Fragment()
    #                 tbegin = timeit.default_timer()
    #                 if not done:
    #                     fragment.store( {"feature_a": torch.Tensor(state)}, outputs['action'], reward, logp, action_mask, 
    #                            1 if done else 0 , outputs['value'], outputs['logits'])
    #             if done:
    #                 # print("total_reward is", total_reward)
    #                 self.total_rewards.append(total_reward)
    #                 break
    #             state = nstate 


    def set_weight(self, weight):
        self.model_dict['weight'] = auto_move(weight,'cpu')
        
    # def update_weight(self):
    #     if 'weight' in self.model_dict:
    #         self.policy._network.load_state_dict(self.model_dict['weight'])
    #     else:
    #         print("now , no weight to update")
    #     self.need_update_weight = False
    #     self.total_rewards[:] = []

    # def init_env(self):
    #     # print(f"Process {os.getpid()} is sampling...")
    #     print("subProcess init env")
    #     self.env = gym.make('CartPole-v1')
    #     self.policy = construct(self.delayed_policy)
    #     # self.policy = self.model_dict['policy']
    #     self.need_update_weight = True
    #     # print(f'Process {os.getpid()} my policy is ', self.policy)
    #     # time.sleep(10)
    #     # self._network = deepcopy(self.model[0])
    
    # @classmethod
    # def worker(cls, delayed_policy, total_rewards, data, data_size, sampling_flag, weights_dict):
    #     import copy
    #     print("sub process finished init env")
    #     policy = construct(delayed_policy)
    #     sampler = cls.sample(policy, total_rewards, data, data_size)
    #     need_update_weight = False
    #     while 1:
    #         try:
    #             if sampling_flag.value:
    #                 if need_update_weight:
    #                     if 'weight' in weights_dict:        
    #                         state_dict = auto_move(weights_dict['weight'], 'cuda')
    #                         policy._network.load_state_dict(state_dict)
    #                         need_update_weight = False
    #                         total_rewards[:] = []
    #                 # print("sample a fragment")
    #                 next(sampler)
    #                 # self.data.append(result)
    #             else:
    #                 time.sleep(1)  # 如果不进行采样，那么就让进程睡眠
    #                 need_update_weight = True
    #         except Exception as e:
    #             print(f"worker Error : {e}")
    #             raise e

    def start_sampling(self):
        print(f"Main Process {os.getpid()} start sampling...")
 
        self.sampling_flag.value = 1
        # self.worker()
        
        for _ in range(self.num_processes):
            ctx = multiprocessing.get_context('spawn')
            p = ctx.Process(target=sample_target, args=(
                {
                    "class": SingleSampler,
                    "params": {
                        'delayed_policy': self.delayed_policy,
                        'total_rewards': self.total_rewards, 
                        'datas': self.datas, 
                        'data_size': self.data_size, 
                        'sampling_flag': self.sampling_flag, 
                        'weights_dict': self.model_dict
                    }
                },
            ))
            p.daemon=True
            p.start()
            # self.processes.append(p)
        print("start sampling over")

    def stop_sampling(self):
        self.sampling_flag.value = 0


class EmptyClass:
    pass

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    self = EmptyClass()
    from lib.network import *
    from lib.PPOPolicy import PPOPolicy
    network_cfg = {
        "encoder_demo" : {
            "class": CommonEncoder,
            "params": {
                "in_features" : 4, 
                "hidden_layer_sizes": [256, 128],
                # "out_features": 64,
            },
            "inputs": ['feature_a']
        },
        "aggregator": {
            "class": DenseAggregator,
            "params": {
                "in_features" : 128, 
                "hidden_layer_sizes": [256, 128],
                "output_size": 256
            },
            "inputs": ['encoder_demo']
        },
        "value_app": {
            "class": ValueApproximator,
            "params": {
                "in_features" : 256, 
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ['aggregator']
        },
        "action" : {
            "class": CategoricalDecoder,
            "params": {
                "n" : 2,
                "hidden_layer_sizes": [256, 128],
            },
            "inputs": ['aggregator']
        }
    }
    delayed_policy = {"class": PPOPolicy, "params": {"policy_config": network_cfg}}
    self.delayed_policy = delayed_policy
    mgr = multiprocessing.Manager()
    self.total_rewards = mgr.list()
    self.datas = mgr.list()
    self.model_dict = mgr.dict()
    self.data_size = multiprocessing.Value('i', 0)
    self.sampling_flag = multiprocessing.Value('i', 1)
    
    start_config = {
        "class": SingleSampler,
        "params": {
            'delayed_policy':   self.delayed_policy,
            'total_rewards':    self.total_rewards, 
            'datas':            self.datas, 
            'data_size':        self.data_size, 
            'sampling_flag':    self.sampling_flag, 
            'weights_dict':     self.model_dict
        }
    }
    # sample_target(start_config)
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=sample_target, args=(start_config,))
    # p = multiprocessing.Process(target=sample_target, args=(start_config,))
    p.daemon=True
    p.start()
    while True:
        time.sleep(5)
        print(f"sample config is {(self.sampling_flag.value)} {(self.data_size.value)}", flush=True)
        # print(f"the sum of frament is {self.data_size.value}", end='/r', flush=True)
