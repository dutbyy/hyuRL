from __future__ import annotations

import pickle
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import drill
import numpy as np

if TYPE_CHECKING:
    from drill.builder import Builder


class EnvironmentDescription:

    def __init__(self, builder: Builder, extra_info):
        self.environment_creator_user_args = {'builder': builder, 'episode_mode': False, 'extra_info': extra_info}
        self.node_id = 0
        self.num_envs_on_this_node = 10
        self.num_envs_on_this_actor = 10
        self.environment_id_on_this_node = 1
        self.environment_id_on_this_actor = 1
        self.environment_id_on_this_task = 1


def add_batch(input_dict: Dict):
    # 给输入中的所有np.ndarray添加一维，变成batch size为1的np.ndarray
    # example: shape (dim0, dim1) -> shape (1, dim0, dim1)
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Dict):
            output_dict[key] = add_batch(value)
        else:
            output_dict[key] = value[np.newaxis, ...]
    return output_dict


def remove_batch(input_dict: Dict):
    # `add_batch`的逆过程
    # example: shape (1, dim0, dim1) -> shape (dim0, dim1)
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Dict):
            output_dict[key] = remove_batch(value)
        else:
            output_dict[key] = np.squeeze(value, axis=0)
    return output_dict


def stack_over_steps(input_dict: Dict):
    # example: {"x": [array([1, 2]), array([2, 3])]} -> {"x": array([[1, 2], [2, 3]])}
    #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^             ^^^^^^^^^^^^^^^^^^^^^^
    #               a list of np.ndarray with shape (2,)     np.ndarray with shape (2, 2)
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Dict):
            output_dict[key] = stack_over_steps(value)
        else:
            output_dict[key] = np.stack(value, axis=0)
    return output_dict


def append_merge(target: Dict, candidate: Dict):
    # example
    # target: {"x": [array([1, 2])]} candidate: {"x": array([2, 3])}
    #                              \/
    #         target: {"x": [array([1, 2]), array([2, 3])]}
    for k, v in candidate.items():
        if isinstance(v, Dict):
            if k not in target:
                target[k] = {}
            append_merge(target[k], v)
        else:
            if k not in target:
                target[k] = []
            target[k].append(v)


def nsteps_to_piece(nsteps):
    flow_states = {}
    flow_actions = {}
    flow_valid_actions = {}
    flow_advs = {}

    for step in nsteps:
        flow_state, flow_action, flow_valid_action, adv = step
        append_merge(flow_states, flow_state)
        append_merge(flow_actions, flow_action)
        append_merge(flow_valid_actions, flow_valid_action)
        append_merge(flow_advs, adv)

    flow_states = stack_over_steps(flow_states)
    flow_actions = stack_over_steps(flow_actions)
    flow_valid_actions = stack_over_steps(flow_valid_actions)
    flow_advs = stack_over_steps(flow_advs)

    return [flow_states, flow_actions, flow_valid_actions, flow_advs]


def for_each_value_of_dict(inputs_dict, callback):
    outputs_dict = {}
    for name, inputs in inputs_dict.items():
        if isinstance(inputs, dict):
            for_each_value_of_dict(inputs, callback)
        else:
            callback(inputs)
    return outputs_dict


def check_state(values):
    assert 'obs' in values
    assert 'model' in values
    assert 'reward' in values['obs']

    def check_for_nested_dict(inputs_dict):
        for _, value in inputs_dict.items():
            if isinstance(value, dict):
                check_for_nested_dict(value)
            else:
                assert isinstance(value, np.ndarray), f"Expect np.ndarray, but get {type(value)}"

    check_for_nested_dict(values['obs'])


def calc_stats(data: Dict[str, np.ndarray]) -> Dict:
    res = {}
    for key, value in data.items():
        tmp = value.ravel()
        res[key] = {
            "count": len(tmp),
            "max": np.max(tmp),
            "min": np.min(tmp),
            "mean": np.mean(tmp),
            "std": np.std(tmp),
            "median": np.median(tmp),
            "75%": np.percentile(tmp, 75),
            "90%": np.percentile(tmp, 90),
        }
    return res


class LocalRunner:

    def __init__(self, flow_model_dict, flow_env):
        self.flow_env = flow_env
        self.flow_env.reset()

        self.names_to_model_instance = flow_model_dict

        for flow_model in self.names_to_model_instance.values():
            # 测试 flow_model 能够通过 pickle 序列化和反序列化
            flow_model_pickled = pickle.dumps(flow_model)
            pickle.loads(flow_model_pickled)

            state = flow_model.__getstate__()
            flow_model.setstate_predict(state)
            flow_model.setstate_learn(state)

    def debug(self, agent_name_to_model_name, fragment_size: int = 128 + 1, return_stats=False):
        self.flow_env.reset()
        states = self.flow_env.observe()

        nsteps_dict = defaultdict(list)

        while states:
            if all(
                    len(nsteps_dict[agent_name]) >= fragment_size
                    for agent_name in agent_name_to_model_name.keys()):
                break

            for agent_name, state in states.items():
                model_name = agent_name_to_model_name[agent_name]
                flow_model = self.names_to_model_instance[model_name]

                check_state(state)
                batch_states = add_batch(state['obs'])
                flow_action = flow_model.predict(batch_states)
                flow_action = remove_batch(flow_action)
                flow_valid_action = self.flow_env.step(agent_name, flow_action)
                if len(nsteps_dict[agent_name]) < fragment_size:
                    flow_state_and_action = []
                    flow_state_and_action.append(state['obs'])
                    flow_state_and_action.append(flow_action)
                    flow_state_and_action.append(flow_valid_action)
                    nsteps_dict[agent_name].append(flow_state_and_action)
            states = self.flow_env.observe()
            print(states)

        assert len(list(nsteps_dict.keys())) > 0

        stats = {} if return_stats else None
        for agent_name, nsteps in nsteps_dict.items():
            self.flow_env.enhance_fragment(agent_name, nsteps)
            piece = nsteps_to_piece(nsteps)
            if return_stats:
                stats[agent_name] = calc_stats(piece[0])
            model_name = agent_name_to_model_name[agent_name]
            flow_model = self.names_to_model_instance[model_name]
            print('learning!!!!!!!!!!!!')
            flow_model.learn(piece)

        print(
            "\033[92m There is no problem with the local detection of your code, feel free to run it in the cloud. \033[0m"
        )
        return stats

    def evaluate(self,
                 agent_name_to_model_name,
                 max_step_num: int = 128,
                 is_render=False,
                 is_batch_inference=True,
                 **render_kwargs):
        self.flow_env.reset()
        states = self.flow_env.observe()
        obs_ = []
        for _ in range(max_step_num):
            if is_render:
                obs = self.flow_env.render(**render_kwargs)
                obs_.append(obs)
            if not states:
                self.flow_env.reset()
                states = self.flow_env.observe()
            flow_action = {}
            # 在多智能体环境中，是否需要将 states 组成 batch 一齐推断出 action
            if is_batch_inference:
                model_to_agent_states = defaultdict(dict)
                for agent_name, state in states.items():
                    model_name = agent_name_to_model_name[agent_name]
                    # 对 states 添加一个维度，方便组成 batch 放在神经网络里面进行训练
                    model_to_agent_states[model_name][agent_name] = state['obs']
                # 每个 model 对应一个 batch_states
                for model_name, agent_name_to_states in model_to_agent_states.items():
                    action = self.batch_inference(model_name, agent_name_to_states)
                    flow_action.update(action)
            else:
                # 多智能体竞争环境：可能有些 agent 死了，所以 agent names 的值要动态获取
                for agent_name, state in states.items():
                    model_name = agent_name_to_model_name[agent_name]
                    flow_model = self.names_to_model_instance[model_name]
                    batch_states = add_batch(state['obs'])
                    action = flow_model.predict(batch_states)
                    action = remove_batch(action)
                    flow_action.update({agent_name: action})

            for agent_name in states.keys():
                self.flow_env.step(agent_name, flow_action[agent_name])
            states = self.flow_env.observe()

        print("\033[92m evaluation done! \033[0m")
        return obs_ if is_render else None

    def batch_inference(self, model_name, agent_name_to_states):
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

        flow_model = self.names_to_model_instance[model_name]
        agent_names = list(agent_name_to_states.keys())
        states = list(agent_name_to_states.values())
        # 将所有 agent 的 states 信息组成一个batch
        batch_states = convert_to_batch_state(states)
        # 在当前 model 下进行 batch predict
        batch_outputs = flow_model.predict(batch_states)
        split_outputs = self._split_output(batch_outputs)

        # 每个 agent 对应各自的 action_dict
        flow_action = {}
        for (agent_name, split_output) in zip(agent_names, split_outputs):
            flow_action[agent_name] = split_output
        return flow_action

    def _split_output(self, inputs):
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
                inner_outputs = self._split_output(value)
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

    def predict(self, model_name_to_batch_states):
        return {
            model_name: self.names_to_model_instance[model_name].predict(states)
            for model_name, states in model_name_to_batch_states.items()
        }


class BPLocalRunner(LocalRunner):

    def __init__(self, builder: Builder, model_info_dict=None, algorithm=None):
        from drill.flow.flow_env_ppo import FlowEnvPPO
        from drill.flow.flow_model_ppo import FlowModelPPO
        drill.local_run = True
        if not algorithm:
            flow_model_func = FlowModelPPO
            flow_env_func = FlowEnvPPO
        else:
            flow_model_func = algorithm['flow_model']
            flow_env_func = algorithm['flow_env']

        # 在用 flow 起训练时，builder 作为 `Actor.start_generate_replays` 的参数 environment_creator_user_args 传入，
        # 需要保证 builder 是可序列化的，参考 https://docs.inspir.work/flow/flow/tutorial.html#flow.api.Actor.start_generate_replays
        builder_pickled = pickle.dumps(builder)
        pickle.loads(builder_pickled)

        # 在用 flow 起训练时，FlowEnv 作为 `Actor.start_generate_replays` 的参数 environmen_creator 传入，
        # 需要保证 FlowEnv 这个 class 本身可序列化，只需 FlowEnv 定义在 module 的 top level
        # 参考 https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
        FlowEnv_pickled = pickle.dumps(flow_env_func)
        pickle.loads(FlowEnv_pickled)

        self.builder = builder
        environment_description = EnvironmentDescription(builder, extra_info={'index': 'training'})
        flow_env = flow_env_func(environment_description)
        flow_env.reset()

        model_names = builder.model_names
        names_to_model_instance = {}

        for model_name in model_names:
            flow_model = flow_model_func(model_name, builder)
#             para_info = {
#                 'trainableParams': np.sum([np.prod(v.get_shape()) for v in flow_model._model._network.trainable_weights]),
#                 'nonTrainableParams': np.sum(
#                     [np.prod(v.get_shape()) for v in flow_model._model._network.non_trainable_weights]),
#             }
#             print(f'Model: {model_name}, Parameters: {para_info}')
            if model_info_dict and model_name in model_info_dict:
                info_dict = model_info_dict[model_name]
                flow_model.load_weights(info_dict["model_file"], info_dict.get("backend",'tensorflow'), info_dict.get("mode",'npz'))
            names_to_model_instance[model_name] = flow_model

        self.agent_name_to_model_name = {
            agent_name: builder.get_model_name(agent_name) for agent_name in builder.agent_names
        }
        super().__init__(names_to_model_instance, flow_env)

    def debug(self, fragment_size: int = 128 + 1, return_stats=False):
        return super().debug(self.agent_name_to_model_name, fragment_size, return_stats)

    def evaluate(self, max_step_num: int = 16, is_render=False, **render_kwargs):
        return super().evaluate(self.agent_name_to_model_name, max_step_num, is_render,
                                **render_kwargs)