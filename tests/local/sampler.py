from __future__ import annotations

import time
import pickle
import logging
from collections import defaultdict
from typing import Dict, Any

from hyurl.flow.drill_plugin.api.flow_api import EnvironmentDescriptor
from hyurl.tools.common import timer_decorator, fix_print
from hyurl.tools.common import Summary

mean = lambda x: sum(x)/len(x)

class LocalSampler:
    def __init__(self, flow_config):
        self.flow_config = flow_config
        self.flow_env = None
        self.flow_models = {}
        self.logger = logging.getLogger("Sampler")
        self.env_desc = EnvironmentDescriptor(1, 1, 1, 0, 0, 0, {
            "builder": self.flow_config['builder'],
            "extra_info": {}
        })

    def reset(self):
        self.flow_env = self.flow_config['algorithm']['flow_env'](self.env_desc)
        self.flow_env.reset()
        self.flow_models = {}

    def get_batch(self, batch_size=512):
        fragments = defaultdict(lambda: [])
        datas = []
        total_rewards = []
        data_size = 0
        fragment_size = 257

        while data_size < batch_size:
            state_dict = self.flow_env.observe()
            if not state_dict:
                self.flow_env.reset()
                state_dict = self.flow_env.observe()

            episode_done = False
            while not episode_done:
                action_dict = {}
                for agent_name, state in state_dict.items():
                    model_name = state['model']
                    if model_name not in self.flow_models:
                        self.flow_models[model_name] = self.flow_config['algorithm']['flow_model'](model_name, self.flow_config['builder'])
                        state_tuple = self.flow_models[model_name].__getstate__()
                        self.flow_models[model_name].setstate_predict(state_tuple)
                    outputs = self.flow_models[model_name].predict(state['obs'])
                    action_dict[agent_name] = outputs

                decoder_mask_dict = {
                    agent_name: self.flow_env.step(agent_name, agent_command_dict)
                    for agent_name, agent_command_dict in action_dict.items()
                }
                nstate_dict: Dict[str, Dict[str, Any]] = self.flow_env.observe()

                if self.flow_env.reseted is not None:
                    total_rewards.append(self.flow_env.reseted)
                    self.flow_env.reseted = None

                piece = [state_dict, action_dict, decoder_mask_dict]
                episode_done = False if nstate_dict else True

                for agent_name in state_dict.keys():
                    agent_piece = [piece[0][agent_name]['obs'], piece[1][agent_name], piece[2][agent_name]]
                    fragments[agent_name].append(agent_piece)

                for agent_name in state_dict.keys():
                    agent_fragments = fragments[agent_name]
                    if len(agent_fragments) >= fragment_size or episode_done:
                        data_size += len(agent_fragments) - 1
                        self.flow_env.enhance_fragment(agent_name, agent_fragments)
                        datas.extend([pickle.dumps(item) for item in agent_fragments])
                        fragments[agent_name].clear()

                        if not episode_done:
                            agent_piece = [piece[0][agent_name]['obs'], piece[1][agent_name], piece[2][agent_name]]
                            fragments[agent_name].append(agent_piece)

                state_dict = nstate_dict

        if len(total_rewards) > 100:
            total_rewards = total_rewards[-100:]
        if len(total_rewards):
            self.logger.info(f"average episode reward is {mean(total_rewards):.1f}")
            Summary.add_scaler('episode_reward', mean(total_rewards))

        return [pickle.loads(item) for item in datas]
