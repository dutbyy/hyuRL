import os
from flow_env_ppo_sync import FlowEnvPPOSync as FlowEnvPPO
from flow_model_ppo_sync import FlowModelPPOSync
import numpy as np
import time
from typing import Dict
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class EnvDescribe:
    def __init__(self, builder, extra_info = {}):
        self.environment_creator_user_args = {'builder': builder, 'extra_info': extra_info, 'episode_mode':False}
        self.node_id = 0
        self.num_envs_on_this_node = 10
        self.num_envs_on_this_actor = 10
        self.environment_id_on_this_node = 1
        self.environment_id_on_this_actor = 1
        self.environment_id_on_this_task = 1

def remove_batch(input_dict):
    output_dict = {
        key : remove_batch(value) if isinstance(value, Dict) else np.squeeze(value, axis=0)
        for key, value in input_dict.items()
    }
    return output_dict

def add_batch(input_dict):
    # print(f"add batch : {input_dict}")
    output_dict = {
        key : add_batch(value) if isinstance(value, Dict) else  np.array(value)[np.newaxis, ...]
        for key, value in input_dict.items()
    }
    return output_dict

def init_model_env(builder, model_path_dict = {}):
    flow_env = FlowEnvPPO(EnvDescribe(builder))
    print("init flow env over")

    print('constructing model')
    print(f"builder.model_names : {builder.model_names}")
    model_dict = {
        model_name: FlowModelPPOSync(model_name, builder)
        for model_name in builder.model_names
    }


    print("init flow model ppo over")

    name2model = {
        agent_name : model_dict.get(builder.get_model_name(agent_name))
        for agent_name in builder.agent_names
    }
    for model_name, model_path in model_path_dict.items():
        print(f"准备加载模型, {model_name}")
        flow_model = name2model.get(model_name, None)
        if flow_model:
            print(f"加载模型 : {model_path}")
            flow_model.load_weights(model_path, 'tensorflow', 'npz')

    return flow_env, name2model

def inference(flow_env, name2model):
    while True:
        flow_env.reset()
        while True:
            states = flow_env.observe()
            if not states:
                print("states is None, episode over!\n\n")
                break
            flow_action = {}
            t0 = time.time()
            if 1:
                time.sleep(.01)
                for agent_name, state in states.items():
                    flow_model = name2model.get(agent_name, None)
                    batch_states = add_batch(state['obs'])
                    action = flow_model.predict(batch_states)
                    action = remove_batch(action)
                    flow_action.update({agent_name: action})
            t1 = time.time()
            # print(f'model predict eplased time : [{round(t1-t0, 2)* 1000} ms]')

            t0 = time.time()
            for agent_name in states.keys():
                flow_env.step(agent_name, flow_action[agent_name])
            t1 = time.time()
            # print(f'env inference eplased time : [{round(t1-t0, 2)* 1000} ms]')


def main():
    model_path_dict = {}
    builder = None
    flow_env, name2model = init_model_env(builder, model_path_dict)
    inference(flow_env, name2model)
