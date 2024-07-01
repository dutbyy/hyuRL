from email import policy
from math import radians
import multiprocessing.shared_memory
import timeit
import gym
from lib.memory.buffer import Memory, Fragment
import torch
from multiprocessing import Pool

from local.sampler import Sampler
from local.trainer import LocalTrainer

from lib.PPOPolicy import PPOPolicy

class Controller:
    def __init__(self) -> None:
        pass
    
    
    def print_info(self, train_step=0):
        print(f"trainning {train_step}")
        pass
    
    
    def train_run(self, policy_config):
        delayed_policy = {"class": PPOPolicy, "params": {"policy_config": policy_config, 'device': 'cuda' }}
        sampler = Sampler(delayed_policy=delayed_policy, num_processes=1)
        delayed_policy = {"class": PPOPolicy, "params": {"policy_config": policy_config, 'device': 'cuda', "trainning": True}}
        trainer = LocalTrainer(delayed_policy=delayed_policy)
        sampler.start_sampling()
        train_step = 0
        while True:
            train_step += 1
            fragements = sampler.get_batch(512)
            my_states = {}
            my_actions = {}
            my_logits = {}
            my_masks = {}
            my_log_probs = {}
            my_advantages = []
            my_is_terminals = []
            my_values = []
            my_size = 0 
            def trans(lis, dic):
                for ilis in lis:
                    for k, v in ilis.items():
                        if k not in dic:
                            dic[k] = []
                        dic[k].append(v)
            for fragment in fragements:
                states, actions, logits, masks, logps, values,  dones, advs = fragment.gens()
                trans(states,   my_states)
                trans(actions,  my_actions)
                trans(logits,   my_logits)
                trans(masks,    my_masks)
                trans(logps,    my_log_probs)
                
                my_values.extend(values)
                my_is_terminals.extend(dones)
                my_advantages.extend(advs)
                my_size = len(my_values)
                
            probabilities = torch.ones(len(my_values))  # 假设所有元素被选中的概率都相同
            batch_indices = torch.multinomial(probabilities, len(my_values), replacement=False)

            train_data = {
                'states':       { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in my_states.items() },
                'actions':      { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in my_actions.items() },
                'logits':       { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in my_logits.items() },
                'masks':        { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in my_masks.items() },
                'log_probs':    { k: torch.stack( [torch.as_tensor(v[i]) for i in batch_indices]) for k , v in my_log_probs.items() },
                'advantages':   torch.stack([torch.as_tensor(my_advantages[i]) for i in batch_indices]),
                'values':       torch.stack([torch.as_tensor(my_values[i]) for i in batch_indices]),
            }
            a = timeit.default_timer() * 1000
            trainer.train(train_data)
            b = timeit.default_timer() * 1000
            print(f'train eplased time : {b-a}')
            weights = trainer.get_state_dict()
            sampler.set_weight(weights)
            self.print_info(train_step)
            sampler.sampling_flag.value = 1
            sampler.datas[:] = []
            sampler.data_size.value = 0

            
            
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    from lib.network import *
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
                "hidden_layer_sizes": [256, 512,1024, 128],
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
    controller = Controller()
    controller.train_run(network_cfg)