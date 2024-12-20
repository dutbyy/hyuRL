from __future__ import annotations

from hyuRL.src.flow.drill_plugin.interface.flow_model import FlowModelPPOSync as FlowModelPPO
from hyuRL.src.flow.drill_plugin.interface.flow_env import FlowEnvImp as FlowEnvPPO
import numpy as np
from typing import Dict
from drill.pipeline.interface import ActionData
import copy
import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Union
import time


from drill.keys import (
    ACTION,
    ACTION_MASK,
    ADVANTAGE,
    CRITIC_HIDDEN_STATE,
    DECODER_MASK,
    DONE,
    HIDDEN_STATE,
    LOGITS,
    REWARD,
    DROP_OUT,
)
# from drill.model.tf.network.commander import CommanderNetwork

class EnvDescribe:
    def __init__(self, builder, extra_info={}):
        self.environment_creator_user_args = {
            "builder": builder,
            "extra_info": extra_info,
        }
        self.node_id = 0
        self.num_envs_on_this_node = 10
        self.num_envs_on_this_actor = 10
        self.environment_id_on_this_node = 1
        self.environment_id_on_this_actor = 1
        self.environment_id_on_this_task = 1


# class FlowEnvPPO:
#     def __init__(self, environment_description):
#         builder = environment_description.environment_creator_user_args["builder"]
#         env_id = environment_description.environment_id_on_this_task
#         env_extra_info = {
#             "node_id": environment_description.node_id,
#             "num_envs_on_this_node": environment_description.num_envs_on_this_node,
#             "num_envs_on_this_actor": environment_description.num_envs_on_this_actor,
#             "environment_id_on_this_node": environment_description.environment_id_on_this_node,
#             "environment_id_on_this_actor": environment_description.environment_id_on_this_actor,
#         }
#         env_extra_info.update(
#             environment_description.environment_creator_user_args["extra_info"]
#         )
#         self._episode_mode_bool = environment_description.environment_creator_user_args[
#             "episode_mode"
#         ]
#         self._builder = builder
#         self._env = builder.build_env(env_id, env_extra_info)
#         self._pipeline = builder.build_pipeline()

#         self.flow_env_config = {}
#         self._episode_done = None  # record whether the episode is over
#         self._obs_data = None

#         # 上个 action, logits, value
#         # <key = agent name, value = {logits, action, value}>
#         self._last_hidden_state_dict = defaultdict(dict)
#         # 当前剩余 agents
#         self._agent_names = []
#         # mask history {agent_name: [{decoder_name: mask}]}
#         # self._dynamic_mask_history = defaultdict(list)
#         self._last_agent_to_reward = {}
#         self._last_agent_to_done = {}

#     @property
#     def agent_names(self) -> List[str]:
#         return self._agent_names

#     @property
#     def env(self):
#         return self._env

#     def reset(self):
#         """重置状态，开始一个新的 episode"""
#         self._obs_data = self._env.reset()
#         self._episode_done = False
#         self._last_hidden_state_dict = defaultdict(dict)
#         self._agent_names = []
#         self.to_step_actions = None
#         self._pipeline.reset()

#     def _get_hidden_state(self, agent_name):
#         if HIDDEN_STATE in self._last_hidden_state_dict[agent_name]:
#             predict_output_dict = self._last_hidden_state_dict[agent_name]
#             hidden_state_dict = {HIDDEN_STATE: predict_output_dict[HIDDEN_STATE]}
#             if CRITIC_HIDDEN_STATE in predict_output_dict:
#                 hidden_state_dict[CRITIC_HIDDEN_STATE] = predict_output_dict[
#                     CRITIC_HIDDEN_STATE
#                 ]
#             return hidden_state_dict
#         return self._builder.get_initial_state(agent_name)

#     def observe(self) -> Dict[str, Dict[str, Union[Dict, str]]]:
#         if self.to_step_actions is not None:
#             self._obs_data, self._episode_done = self._env.step(self.to_step_actions)
#         self.to_step_actions = {}

#         agent_to_state, agent_to_reward = self._pipeline.pre_process(
#             self._obs_data, self._episode_done
#         )
#         self._last_agent_to_reward.update(agent_to_reward)
#         for agent_name in self.env.agent_names:
#             self._last_agent_to_done[agent_name] = self._episode_done

#         if self._episode_mode_bool:
#             if self._episode_done:
#                 return {}

#         episode_done = self._episode_done
#         if episode_done:
#             self.reset()
#             agent_to_state, _ = self._pipeline.pre_process(
#                 self._obs_data, self._episode_done
#             )

#         observe_return = {}
#         for agent_name, agent_state_dict in agent_to_state.items():
#             # TODO: recording each individual reward target
#             reward = self._last_agent_to_reward[agent_name]
#             if isinstance(reward, dict):
#                 agent_state_dict[REWARD] = np.array(
#                     sum(reward.values()), dtype=np.float32
#                 )
#             else:
#                 agent_state_dict[REWARD] = np.array(reward, dtype=np.float32)
#             agent_state_dict[DONE] = np.array(episode_done, dtype=np.float32)
#             hidden_state_dict = self._get_hidden_state(agent_name)
#             agent_state_dict.update(hidden_state_dict)

#             observe_return[agent_name] = {
#                 "obs": agent_state_dict,
#                 "model": self._builder.get_model_name(agent_name),
#             }

#         self._agent_names = [agent_name for agent_name in observe_return.keys()]

#         return observe_return

#     def step(
#         self, agent_name: str, predict_output: Dict[str, Any]
#     ) -> Dict[str, Dict[str, np.ndarray]]:
#         if HIDDEN_STATE in predict_output:
#             self._last_hidden_state_dict[agent_name][HIDDEN_STATE] = copy.deepcopy(
#                 predict_output[HIDDEN_STATE]
#             )
#             if CRITIC_HIDDEN_STATE in predict_output:
#                 self._last_hidden_state_dict[agent_name][CRITIC_HIDDEN_STATE] = (
#                     copy.deepcopy(predict_output[CRITIC_HIDDEN_STATE])
#                 )

#         data = ActionData(
#             action=copy.deepcopy(predict_output[ACTION]),
#             predict_output=copy.deepcopy(predict_output),
#         )

#         action_dict, action_mask, decoder_mask = self._pipeline.post_process(
#             agent_name, data
#         )

#         if action_dict is not None:
#             # self._obs_data, self._episode_done = self._env.step(action_dict)
#             self.to_step_actions.update(action_dict)

#         # FIXME: change this return once Flow implement a newer version
#         return {DECODER_MASK: decoder_mask}

#     def enhance_fragment(self, agent_name: str, fragments: List[Dict]):
#         """对不断 `observe` 和 `step` 收集的数据进行处理，这里计算了 GAE

#         什么时候调用这个方法？
#         一次 `observe` 和 `step` 收集的数据记为一个 fragment，当收集到的数据达到
#         `fragment_size` （一个配置参数）时调用此方法

#         注意: fragments 只能原地修改，这个方法不接受返回值，这是由 flow 决定的

#         Parameters
#         ----------
#         agent_name : str
#             agent 的名字
#         fragments : List[Dict]
#             长度为 `fragment_size`，每一个元素都是 3 元组，分别对应 `observe` 的
#             返回值（准确的说是 `observe_return[agent_name]["obs"]`，`step` 的参数
#             `predict_output` 和 `step` 的返回值。
#         """

#         # 对于多智能体竞争环境存在一种情况：
#         # 采集 fragment 的顺序：先依次执行FlowEnv.step() 和 FlowEnv.observe() 得
#         # 到 state, reward, 然后执行 FlowModel.predict() 得到 action, value，然后采集
#         # 一个 fragment(state, reward, action, value)，然后执行下一个 FlowEnv.step()。
#         # 设 fragment_size=64+1 = 65，第 64 次执行 FlowEnv.step() 时，已
#         # 经采集了 64 个fragment（若某个 agent 在第 64 次 step done了，返回的
#         # obs, reward, done 不为空），开始采集第 65 个fragment，然后会执行第 65 个
#         # FlowEnv.step(), 再执行 FlowEnv.enhance_fragment()，此时进入 enhance_fragment()
#         # 后会遇到问题: 此时的 self._env_info 已经不包含刚刚 done 的 agent 了，因此 last_reward
#         # = None。因此额外使用 self._final_reward 维护所有 done 的 agent 的最后一个 reward。

#         rewards = []
#         values = []
#         dones = []
#         drop_outs = []

#         for i in range(len(fragments)):
#             flow_state_dict, flow_action_dict, _ = fragments[i]
#             values.append(flow_action_dict["value"])
#             rewards.append(flow_state_dict["reward"])
#             dones.append(flow_state_dict["done"])
#             if DROP_OUT in flow_state_dict:
#                 drop_outs.append(flow_state_dict[DROP_OUT])
#             else:
#                 drop_outs.append(0)
#             # fragments[i][-1][ACTION_MASK] = self._dynamic_mask_history[agent_name][i]

#         if self._episode_mode_bool:
#             rewards.append(
#                 np.array(
#                     self._last_agent_to_reward[agent_name]["reward"], dtype=np.float32
#                 )
#             )
#             dones.append(self._last_agent_to_done[agent_name])
#             values.append(np.array(0.0, dtype=np.float32))
#         advantages = self._pipeline.batch_process(agent_name, rewards, values, dones)

#         # 算好advantage后将其append到fragments中
#         # 这一块也比较难理解，advantage的长度还能和fragments不一样？
#         # 先来看看advantage的计算，adv = r_t + gamma * v_(t+1) - v_t
#         # 主要原因就是我们在算在s_t下执行a_t的advantage的时候，需要用到下一步v值。
#         # 假设fragment的长度是n的话，我们能计算出的advantage数量只有n-1。
#         fragments_len = len(fragments)
#         advantages_len = len(advantages)

#         for i in range(advantages_len):
#             fragments[i].append(
#                 {ADVANTAGE: np.asarray(advantages[i], dtype=np.float32)}
#             )
#         if fragments_len == advantages_len + 1:
#             fragments.pop(-1)
#             # del self._dynamic_mask_history[agent_name]

#         for i in reversed(range(len(fragments))):
#             if drop_outs[i] == 1:
#                 fragments.pop(i)

#     def render(self, **kwargs):
#         if hasattr(self._env, "render"):
#             return self._env.render(**kwargs)  # type: ignore
#         else:
#             raise RuntimeError("self._env does not have a render method!")


remove_batch = lambda input_dict : {
    key: (remove_batch(value) if isinstance(value, Dict) else np.squeeze(value, axis=0))
    for key, value in input_dict.items()
}

add_batch = lambda input_dict : {
    key: ( add_batch(value) if isinstance(value, Dict) else np.array(value)[np.newaxis, ...])
    for key, value in input_dict.items()
}

# def remove_batch(input_dict):
#     output_dict = {
#         key: (
#             remove_batch(value)
#             if isinstance(value, Dict)
#             else np.squeeze(value, axis=0)
#         )
#         for key, value in input_dict.items()
#     }
#     return output_dict

# def add_batch(input_dict):
#     # print(f"add batch : {input_dict}")
#     output_dict = {
#         key: (
#             add_batch(value)
#             if isinstance(value, Dict)
#             else np.array(value)[np.newaxis, ...]
#         )
#         for key, value in input_dict.items()
#     }
#     return output_dict


def init_env(builder, env_id=1):
    desc = EnvDescribe(builder)
    desc.environment_id_on_this_node = env_id
    desc.environment_creator_user_args["episode_mode"] = True
    flow_env = FlowEnvPPO(desc)
    return flow_env

def init_model(builder, model_path_dict={}):
    model_dict = {
        model_name: FlowModelPPO(model_name, builder)
        for model_name in builder.model_names
    }

    for model_name, model_path in model_path_dict.items():
        flow_model = model_dict.get(model_name, None)
        if flow_model:
            try:
                flow_model.load_weights(model_path, "tensorflow", "npz")
                print(f"Loading Model {model_name} : {model_path}")
            except Exception as e:
                print(f"loading Model {model_name}: {model_path} failed : {e}")

            if True:
                def make_fake_inputs(flow_model: FlowModelPPO):
                    network = flow_model._model.network
                    encoders = network._encoder_config
                    fake_inputs = {
                        feature_set.name: np.random.normal(size=(1, *feature_set.shape)).astype(np.float32)
                        for feature_set in [encoder.get("inputs") for encoder in encoders.values()]
                    }
                    fake_inputs.update({"reward": np.ones(1, dtype=np.float32), "done": np.zeros(1, dtype=np.float32),})
                    return fake_inputs
                flow_model.predict(make_fake_inputs(flow_model))

    name2model = {
        agent_name: model_dict.get(builder.get_model_name(agent_name))
        for agent_name in builder.agent_names
    }

    return name2model

def init_model_env(builder, model_path_dict={}):
    name2model = init_model(builder, model_path_dict)
    flow_env = init_env(builder=builder)
    return flow_env, name2model

def split_output(inputs):
    """
    将 batch_output 分成每个 agent_name 对应的数据
    Example:
        inputs: {
            "logits": {
                "move": [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]]
            },
            "action": {
                "move": [2, 3, 0]
            }
        }
        outputs: [
            {"logits": { "move": [0, 0, 1, 1] }, "action": {"move": 2}},
            {"logits": { "move": [0, 0, 1, 1] }, "action": {"move": 3}},
            {"logits": { "move": [1, 1, 0, 0] }, "action": {"move": 0}}
        ]
    """
    outputs = []
    for key, value in inputs.items():
        if isinstance(value, dict):
            inner_outputs = split_output(value)
            for i, inner_output in enumerate(inner_outputs):
                if i < len(outputs):
                    outputs[i][key] = inner_output
                else:
                    outputs.append({key: inner_output})
        else:
            rows = np.split(value, len(value))
            for i, row in enumerate(rows):
                row = np.squeeze(row)
                if i < len(outputs):
                    outputs[i][key] = row
                else:
                    outputs.append({key: row})
    return outputs

def batch_inference(flow_model, agent_name_to_states):
    """
    每个agent都有一个model，当agents数量很多时，每次循环都要inference，这样会出现问题
    所以我们需要得到所有的states再一齐inference
    """

    def convert_to_batch_state(states: List[Dict]):
        assert len(states) > 0
        batch_state_dict = {}
        for key, state in states[0].items():
            if isinstance(state, dict):
                sub_states = [s[key] for s in states]
                batch_state_dict[key] = convert_to_batch_state(sub_states)
            else:
                batch_state_dict[key] = np.stack([s[key] for s in states])
        return batch_state_dict

    agent_names = list(agent_name_to_states.keys())
    states = list(agent_name_to_states.values())
    # 将所有 agent 的 states 信息组成一个batch
    batch_states = convert_to_batch_state(states)
    # 在当前 model 下进行 batch predict
    batch_outputs = flow_model.predict(batch_states)
    split_outputs = split_output(batch_outputs)
    # 每个 agent 对应各自的 action_dict
    flow_action = {}
    for (agent_name, split_output_data) in zip(agent_names, split_outputs):
        flow_action[agent_name] = split_output_data
    return flow_action

def inference(flow_env, name2model, episode_num=10):
    avg_infer_times = []
    for i in range(episode_num):
        flow_env.reset()
        while True:
            if flow_env._episode_done:
                break
            states = flow_env.observe()

            flow_action = {}
            # single inference for range
            '''
            for agent_name, state in states.items():
                if not state["obs"]:
                    print(f"跳过 {agent_name}")
                    continue
                flow_model: FlowModelPPO = name2model.get(agent_name, None)
                batch_states = add_batch(state["obs"])
                a = time.time()
                action = flow_model.predict(batch_states)
                b = time.time()
                action = remove_batch(action)
                flow_action.update({agent_name: action})
                print(f"model {agent_name} inference eplased : {b*1000-a*1000} ms")
            '''
            model_to_agent_states = defaultdict(dict)
            model_name2model = {}

            for agent_name, state in states.items():
                if not state['obs']: continue
                flow_model:FlowModelPPO = name2model.get(agent_name, None)
                model_name = flow_model._model_name
                model_to_agent_states[model_name][agent_name] = state['obs']
                model_name2model[model_name] = flow_model

             # 每个 model 对应一个 batch_states
            for model_name, agent_name_to_states in model_to_agent_states.items():
                action = batch_inference(model_name2model[model_name], agent_name_to_states)
                flow_action.update(action)

            avg = lambda x: sum(x)/len(x)
            if len(avg_infer_times) == 1:
                print(f"model inference avg eplased : {avg(avg_infer_times):.2f} ms")
                avg_infer_times.clear()

            for agent_name in states.keys():
                flow_env.step(agent_name, flow_action[agent_name])
