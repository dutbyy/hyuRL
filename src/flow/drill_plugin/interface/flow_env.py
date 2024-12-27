from optparse import Option
import numpy as np
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict
import copy

from hyuRL.src.flow.drill_plugin.api.flow_api import Environment, EnvironmentDescriptor, NestedNDArray

from drill.builder import Builder
from drill.pipeline import ActionData
from drill.keys import HIDDEN_STATE, CRITIC_HIDDEN_STATE, ACTION, DECODER_MASK, REWARD, DONE, DROP_OUT, ADVANTAGE


def getLogger(env_id):
    import logging
    log_name = env_id if isinstance(env_id, str) else f"env-{env_id}"
    logger = logging.getLogger(f"env-{env_id}")
    logger.setLevel(20)
    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    try:
        import os
        os.system("mkdir -p /job/logs/user_log/")
        handler = logging.FileHandler(f"/job/logs/user_log/{log_name}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except:
        pass
    return logger

class FlowEnvImp(Environment):
    """
    创建一个新的环境
    用户需要提供一个environment_creator给Flow, Flow会使用形如 environment_creator(environment_descriptor) 的 调用方式创建Environment Object. 因此如果此类型的__init__方法参数只有environment_descriptor, 可以直接将类型名 作为environment_creator.
    参数
    environment_descriptor (flow.api.EnvironmentDescriptor) –
    """

    def __init__(self, env_desc: EnvironmentDescriptor):
        builder: Builder = env_desc.environment_creator_user_args["builder"]
        env_id = env_desc.environment_id_on_this_task
        env_extra_info = {
            "node_id": env_desc.node_id,
            "num_envs_on_this_node": env_desc.num_envs_on_this_node,
            "num_envs_on_this_actor": env_desc.num_envs_on_this_actor,
            "environment_id_on_this_node": env_desc.environment_id_on_this_node,
            "environment_id_on_this_actor": env_desc.environment_id_on_this_actor,
        }
        env_extra_info.update(env_desc.environment_creator_user_args["extra_info"])
        # self._episode_mode_bool = env_desc.environment_creator_user_args["episode_mode"]

        self._builder = builder
        self._env = builder.build_env(env_id, env_extra_info)
        self._pipeline = builder.build_pipeline()

        self.flow_env_config = {}
        self._episode_done = None
        self._obs_data: Dict[str, Any] = None

        self._last_hidden_state_dict = defaultdict(dict)
        self._agent_names = []
        self._last_agent_to_reward = {}
        self._last_agent_to_done = {}
        self._command_dict = None
        self.logger = getLogger(f"FlowEnv-{env_id}")

    def reset(self) -> None:
        self.logger.info("calling reset")
        """重置状态，开始一个新的 episode"""
        self._pipeline.reset()
        self._obs_data = self._env.reset()
        self._episode_done = False

        self._last_hidden_state_dict = defaultdict(dict)
        self._agent_names = []
        self._last_agent_to_reward = {}
        self._last_agent_to_done = {}
        self._command_dict = {}

    def observe(self) -> Dict[str, NestedNDArray]:
        self.logger.info("calling observe")

        # 如果上次的_obs_data 是空的
        if self._episode_done:
            return {}

        if self._command_dict:
            self._obs_data, self._episode_done = self._env.step(self._command_dict)
            self._command_dict.clear()

        agent2state, agent2reward = self._pipeline.pre_process(self._obs_data, self._episode_done)
        for agent_name, agent_state_dict in agent2state.items():
            reward = agent2reward.get(agent_name)
            if isinstance(reward, dict):
                agent_state_dict[REWARD] = np.array(sum(reward.values()), dtype=np.float32)
            else:
                agent_state_dict[REWARD] = np.array(reward, dtype=np.float32)
            agent_state_dict[DONE] = np.array(self._episode_done, dtype=np.float32)
        observe_return = {
            agent_name: {
                "obs": agent_state_dict,
                "model": self._builder.get_model_name(agent_name),
            }
            for agent_name, agent_state_dict in agent2state.items()
        }

        self._agent_names = list(observe_return.keys())
        return observe_return


    def step(self, agent_name, predict_output): 
        self.logger.info(f"calling step {agent_name}")
        self.__update_hidden_state(agent_name, predict_output)

        action_data = ActionData(
            action=copy.deepcopy(predict_output[ACTION]),
            predict_output=copy.deepcopy(predict_output),
        )
        agent_command_dict, action_mask, decoder_mask_dict = self._pipeline.post_process(agent_name, action_data)
        self._command_dict.update(agent_command_dict)
        return {DECODER_MASK: decoder_mask_dict}

    def enhance_fragment(self, agent_name:str, fragments: List[Any]):

        self.logger.info(f"enhance_fragment {agent_name}")
        rewards = []
        values = []
        dones = []
        drop_outs = []

        # 遍历每个 frame/transition 的数据;
        for i in range(len(fragments)):
            state_dict, action_dict, decoder_mask_dict = fragments[i]
            rewards.append(state_dict["reward"])
            values.append(action_dict["value"])
            dones.append(state_dict["done"])
            if DROP_OUT in state_dict:
                drop_outs.append(state_dict[DROP_OUT])
            else:
                drop_outs.append(0)

        def cal_gae(rewards, values, dones):
            advantages = []
            advantage = 0.0
            gamma: float = 0.99
            lamb: float = 0.0
            for i in reversed(range(len(rewards) - 1)):
                reward, value, next_value = rewards[i + 1], values[i], values[i + 1]
                non_terminate = 1 - int(dones[i + 1])
                delta = reward - (value - gamma * next_value * non_terminate)
                advantage = delta + gamma * lamb * advantage * non_terminate
                advantages.append(advantage)
            return list(reversed(advantages))


        advantages = cal_gae(rewards, values, dones)

        fragments_len = len(fragments)
        advantages_len = len(advantages)
        for i in range(advantages_len):
            fragments[i].append(
                { ADVANTAGE: np.asarray(advantages[i], dtype=np.float32) }
            )
        if fragments_len == advantages_len + 1:
            fragments.pop(-1)

        for i in reversed(range(len(fragments))):
            if drop_outs[i] == 1:
                fragments.pop(i)

    def __update_hidden_state(self, agent_name, predict_output):
        if CRITIC_HIDDEN_STATE in predict_output:
            self._last_hidden_state_dict[agent_name][HIDDEN_STATE] = copy.deepcopy(predict_output[HIDDEN_STATE])
        if CRITIC_HIDDEN_STATE in predict_output:
            self._last_hidden_state_dict[agent_name][CRITIC_HIDDEN_STATE] = copy.deepcopy(predict_output[CRITIC_HIDDEN_STATE])

    @property
    def agent_names(self) -> List[str]:
        return self._agent_names

    def set_config(self, config):
        pass
