from __future__ import annotations

import numpy as np
from typing import Dict
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Union


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

remove_batch = lambda input_dict : {
    key: (remove_batch(value) if isinstance(value, Dict) else np.squeeze(value, axis=0))
    for key, value in input_dict.items()
}

add_batch = lambda input_dict : {
    key: ( add_batch(value) if isinstance(value, Dict) else np.array(value)[np.newaxis, ...])
    for key, value in input_dict.items()
}


def init_env(flow_config, env_id=1):
    builder, flow_env_class = flow_config['builder'], flow_config['algorithm']['flow_env']
    desc = EnvDescribe(builder)
    desc.environment_id_on_this_node = env_id
    desc.environment_creator_user_args["episode_mode"] = True
    flow_env = flow_env_class(desc)
    return flow_env

def init_model(flow_config, model_path_dict={}):
    builder, flow_model_class = flow_config['builder'], flow_config['algorithm']['flow_model']
    model_dict = {
        model_name: flow_model_class(model_name, builder)
        for model_name in builder.model_names
    }
    print(f"model_path_dict is {model_path_dict}")
    for model_name, model_path in model_path_dict.items():
        flow_model = model_dict.get(model_name, None)
        if flow_model:
            flow_model.load_weights(model_path)
            print(f"loding model :{model_path}")

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
    def trans2tensor(nested_structure):
        import tree
        import torch
        # if torch.cuda.is_available():
        #     return tree.map_structure(lambda x: torch.from_numpy(x).cuda(), nested_structure)
        return nested_structure
    batch_states = trans2tensor(batch_states)
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
                flow_model = name2model.get(agent_name, None)
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
