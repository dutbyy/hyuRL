from __future__ import annotations

import copy
from collections import defaultdict
from curses import raw
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np

from drill.keys import (ACTION, ACTION_MASK, ADVANTAGE, CRITIC_HIDDEN_STATE, DECODER_MASK, DONE,
                        HIDDEN_STATE, LOGITS, REWARD, DROP_OUT)
from drill.pipeline import ActionData

if TYPE_CHECKING:
    from drill.builder import Builder
'''
nested np.array是Flow输入输出的主要数据格式, 可以是np.array、list、dict[以str为key]间任意形式的嵌套, 定义如下:
    单个np.array是nested np.array
    list of nested np.array是nested np.array
    以str为key, nested np.array 为 value的dict是np.array
'''
class FlowEnvCommon:
    def __init__(self, environment_description):
        self.env_id         = environment_description.environment_id_on_this_task
        self._builder       = environment_description.environment_creator_user_args['builder']
        self.espidoe_mode   = environment_description.environment_creator_user_args.get('episode_mode', False)
        self.extra_info     = environment_description.environment_creator_user_args['extra_info']
        
        # node_id (int)                         – this_node在cluster中的编号, 关于编号方式见 flow.api.Cluster
        # num_envs_on_this_node (int)           – this_actor在this_node上部署的environments数量
        # num_envs_on_this_actor (int)          – this_actor部署的environments数量
        # environment_id_on_this_node (int)     – 在this_node中所有this_actor部署的environments中此environment的编号, 范围为[0, num_envs_on_this_node - 1]
        # environment_id_on_this_actor (int)    – 在this_actor部署的所有environments中此environment的编号，范围为[0, num_envs_on_this_actor - 1]
        # environment_id_on_this_task (int)     – 在整个任务部署的所有environments中此environment的编号
        # environment_creator_user_args (any)   – 用户在flow.api.Actor.start_generate_replays() 传入的参数

        self.agent_dict = defaultdict(lambda: None)
        self.agent_names = []
        
        self._env = self._builder.build_env()


    def reset(self):
        """开始一个新的episode
        """
        self.agent_dict.clear()
        pass
    
    def observe(self):
        """获取观测数据
        
        Returns:
            observation (): 观测数据, 需要让 model inference 的 agent 的 observation。
            包括网络需要的输入以及对应的 model, 另外还包括 reward 和 done
        Examples:
        ---------
        Returns:
        e.g. return {agent_name: {'obs': obs, 'model': model_name}}
            其中 obs 的类型为 nested np.array, shape和dtype必须固定
        ```python
        observe_return = {
            "red_agent":
                {
                    "obs": {
                        "spatial": np.ndarray,
                        "entity": np.ndarray,
                        "reward": array,
                        "hidden_state": Any,
                    },
                    "model": "battle_model",
                }
            }
            "blue_agent": {}
        }
        ```
        """
        if self.agent_dict:
            raw_observation = self._env.step(self.agent_dict)

        observe_return = {
            k: self.agent_manager[k].obs2state(v)
            for k, v in raw_observation.items()
        }
        return observe_return


    def step(self, agent_name, action):
        """智能体执行动作

        Args:
            agent_name (str): 智能体名称
            action (nested np.array): 神经网络输出的action
        Returns:
            reward (nested np.array): 奖励; 这里为了兼容多智能体并行调用step, 弃用该字段, reward 放在了obs内, 统一在enhance fragment内处理
        """

        action_input = self.agent_dict[agent_name].step(action)
        self.action_dict[agent_name] = action_input
        return 
    
    def enhance_fragment(self, agent_name, fragment):
        """ 
            每次 observe+step 收集到的数据为一个 transition, 一定长度的transitions被收集后, 用户需要调用本方法将它们合并为一个 transitions, 即fragment
            用户需要原地更改输入的fragment, 并保证更改后的fragment类型为 list of enhanced transitions,
            其中enhanced transition 的类型为 nested np.array, 
            且其shape和dtype都必须固定
            什么时候会被调用:
                1. 当一个Episode结束（也就是 flow.api.Environment.observe() 返回空字典）
                2. 当收集到的Transition长度等于 flow.api.ModelDescriptor 中指定的 fragment_size
                3. 当Agent对应的Model发生改变
            调用结束后, 会依据replay_size切割为连续的replay片段(不够的会丢弃)
        Args:
            agent_name (str): 智能体名称
            fragment (list of transitions): 长度在[1, fragment_size]范围内的list, 
                每个元素为一个长度最多为3的 python list [Observation, Action, Reward]
                    Observation (nested np.array): observe方法返回的<agent_name, {"obs": obs, "model": model}>中的obs
                    Action (nested np.array): step方法的参数, 也即agent传回的action
                    Reward (nested np.array): step方法的返回值. 如果step方法返回长度为0的list, 则没有此项
        """

        self.agent_manager[agent_name].enhance_fragment(fragment)
        return 

class FlowEnvPPO:
    """ [flow.api.Environment](https://docs.inspir.work/flow/flow/tutorial.html#flow.api.Environment) 的一个实现，
    负责和 model 交互产生样本

    Attributes
    ----------
    environment_description: flow.api.EnvironmentDescriptor
        云端训练，使用 flow.EnvironmentDescriptor
        本地调试，使用 drill.local.local.EnvironmentDescription
    """

    def __init__(self, environment_description: 'flow.EnvironmentDescriptor'):
        builder: Builder = environment_description.environment_creator_user_args['builder']
        env_id = environment_description.environment_id_on_this_task
        env_extra_info = {'node_id': environment_description.node_id,
                          'num_envs_on_this_node': environment_description.num_envs_on_this_node,
                          'num_envs_on_this_actor': environment_description.num_envs_on_this_actor,
                          'environment_id_on_this_node': environment_description.environment_id_on_this_node,
                          'environment_id_on_this_actor': environment_description.environment_id_on_this_actor}
        env_extra_info.update(environment_description.environment_creator_user_args['extra_info'])
        self._episode_mode_bool = environment_description.environment_creator_user_args['episode_mode']
        self._builder = builder
        self._env = builder.build_env(env_id, env_extra_info)
        self._pipeline = builder.build_pipeline()
        self.flow_env_config = {}
        self._episode_done = None  
        self._obs_data = None
        self._last_hidden_state_dict = defaultdict(dict)
        self._agent_names = []
        self._last_agent_to_reward = {}
        self._last_agent_to_done = {}

    @property
    def agent_names(self) -> List[str]:
        return self._agent_names

    @property
    def env(self):
        return self._env

    # def set_config(self, config):
    #     pass

    def reset(self):
        """ 重置状态，开始一个新的 episode
        """
        self._obs_data = self._env.reset()
        self._episode_done = False
        self._last_hidden_state_dict = defaultdict(dict)
        self._agent_names = []
        self._pipeline.reset()

    def _get_hidden_state(self, agent_name):
        if HIDDEN_STATE in self._last_hidden_state_dict[agent_name]:
            predict_output_dict = self._last_hidden_state_dict[agent_name]
            hidden_state_dict = {HIDDEN_STATE: predict_output_dict[HIDDEN_STATE]}
            if CRITIC_HIDDEN_STATE in predict_output_dict:
                hidden_state_dict[CRITIC_HIDDEN_STATE] = predict_output_dict[CRITIC_HIDDEN_STATE]
            return hidden_state_dict
        return self._builder.get_initial_state(agent_name)

    def observe(self) -> Dict[str, Dict[str, Union[Dict, str]]]:
        """ 获取 observation

        Returns
        -------
        Dict[str, Dict[str, Union[Dict, str]]]
            需要让 model inference 的 agent 的 observation。包括网络需要的输入以及对应的 model，
            另外还包括 reward 和 done

        Examples
        --------
        return
        ```python
        observe_return = {
            "red_agent":
                {"obs":
                    {"spatial": np.ndarray,
                        "entity": np.ndarray,
                        "reward": array,
                        "hidden_state": Any,
                        ...
                    },
                "model": "battle_model"}}
            "blue_agent": ...
        }
        ```
        """
        # 1. 调用 pipeline 将环境返回的状态信息(observation)处理成网络可识别
        # 的 state，reward (环境可能不会返回reward)
        # example state_dict format {agent_name: {fs.name: processed_obs}}
        # example reward_dict format {agent_name: reward}
        agent_to_state, agent_to_reward = self._pipeline.pre_process(self._obs_data,
                                                                     self._episode_done)
        self._last_agent_to_reward.update(agent_to_reward)
        for agent_name in self.env.agent_names:
            self._last_agent_to_done[agent_name] = self._episode_done

        if self._episode_mode_bool:
            if self._episode_done:
                return {}

        episode_done = self._episode_done
        if episode_done:
            self.reset()
            agent_to_state, _ = self._pipeline.pre_process(self._obs_data, self._episode_done)

        observe_return = {}
        for agent_name, agent_state_dict in agent_to_state.items():
            # TODO: recording each individual reward target
            reward = self._last_agent_to_reward[agent_name]
            if isinstance(reward, dict):
                agent_state_dict[REWARD] = np.array(sum(reward.values()), dtype=np.float32)
            else:
                agent_state_dict[REWARD] = np.array(reward, dtype=np.float32)
            agent_state_dict[DONE] = np.array(episode_done, dtype=np.float32)
            hidden_state_dict = self._get_hidden_state(agent_name)
            agent_state_dict.update(hidden_state_dict)

            observe_return[agent_name] = {
                'obs': agent_state_dict,
                'model': self._builder.get_model_name(agent_name)
            }

        self._agent_names = [agent_name for agent_name in observe_return.keys()]

        return observe_return

    def step(self, agent_name: str, predict_output: Dict[str,
                                                         Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """ 根据 model inference 的结果和环境进行交互。
        model 根据 observation 做 inference 得到的结果通过参数 `predict_output` 返回。
        注意：多智能体时，`observe` 可能同时返回所有 agent 的 observation 给 model 做
        inference，但是 `step` 的参数 `predict_output` 只包含参数 `agent_name` 对应的
        inference 结果。且不保证不同 agent step 调用的顺序。

        Parameters
        ----------
        agent_name : str
            agent 的名字
        predict_output : Dict[str, Any]
            `agent_name` 对应的 model inference 的结果
            包含该智能体的 action, logits, value, hidden_state。

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            decoder_mask。复杂一点的场景，action 是有多头的（multi-head）， 而 action 会作为样本收集起来参与训练，
            但并不一定每一个 head 都是 “有效的”。举个例子，假设动作空间为

            * meta: 移动/攻击
            * postion: 目标位置
            * target: 目标单位

            当 meta 为移动的时候，postion 这个 head 是有效的，而 target 无效，所以 target 这个 head 就不应该参与
            loss 计算。decoder mask 的作用就是讲 “无效的” head mask 掉，不参与 loss 计算。

        """
        if HIDDEN_STATE in predict_output:
            self._last_hidden_state_dict[agent_name][HIDDEN_STATE] = copy.deepcopy(
                predict_output[HIDDEN_STATE])
            if CRITIC_HIDDEN_STATE in predict_output:
                self._last_hidden_state_dict[agent_name][CRITIC_HIDDEN_STATE] = copy.deepcopy(
                    predict_output[CRITIC_HIDDEN_STATE])

        data = ActionData(action=copy.deepcopy(predict_output[ACTION]),
                          predict_output=copy.deepcopy(predict_output))
        action_dict, action_mask, decoder_mask = self._pipeline.post_process(agent_name, data)

        if action_dict is not None:
            self._obs_data, self._episode_done = self._env.step(action_dict)
        #     for agent_name, mask in action_mask.items():
        #         self._dynamic_mask_history[agent_name].append(mask)

        # FIXME: change this return once Flow implement a newer version
        return {DECODER_MASK: decoder_mask}

    def enhance_fragment(self, agent_name: str, fragments: List[Dict]):
        """ 对不断 `observe` 和 `step` 收集的数据进行处理，这里计算了 GAE

        什么时候调用这个方法？
        一次 `observe` 和 `step` 收集的数据记为一个 fragment，当收集到的数据达到
        `fragment_size` （一个配置参数）时调用此方法

        注意: fragments 只能原地修改，这个方法不接受返回值，这是由 flow 决定的

        Parameters
        ----------
        agent_name : str
            agent 的名字
        fragments : List[Dict]
            长度为 `fragment_size`，每一个元素都是 3 元组，分别对应 `observe` 的
            返回值（准确的说是 `observe_return[agent_name]["obs"]`，`step` 的参数
            `predict_output` 和 `step` 的返回值。
        """

        # 对于多智能体竞争环境存在一种情况：
        # 采集 fragment 的顺序：先依次执行FlowEnv.step() 和 FlowEnv.observe() 得
        # 到 state, reward, 然后执行 FlowModel.predict() 得到 action, value，然后采集
        # 一个 fragment(state, reward, action, value)，然后执行下一个 FlowEnv.step()。
        # 设 fragment_size=64+1 = 65，第 64 次执行 FlowEnv.step() 时，已
        # 经采集了 64 个fragment（若某个 agent 在第 64 次 step done了，返回的
        # obs, reward, done 不为空），开始采集第 65 个fragment，然后会执行第 65 个
        # FlowEnv.step(), 再执行 FlowEnv.enhance_fragment()，此时进入 enhance_fragment()
        # 后会遇到问题: 此时的 self._env_info 已经不包含刚刚 done 的 agent 了，因此 last_reward
        # = None。因此额外使用 self._final_reward 维护所有 done 的 agent 的最后一个 reward。

        rewards = []
        values = []
        dones = []
        drop_outs = []

        for i in range(len(fragments)):
            flow_state_dict, flow_action_dict, _ = fragments[i]
            values.append(flow_action_dict["value"])
            rewards.append(flow_state_dict['reward'])
            dones.append(flow_state_dict['done'])
            if DROP_OUT in flow_state_dict:
                drop_outs.append(flow_state_dict[DROP_OUT])
            else:
                drop_outs.append(0)
            # fragments[i][-1][ACTION_MASK] = self._dynamic_mask_history[agent_name][i]

        if self._episode_mode_bool:
            rewards.append(np.array(self._last_agent_to_reward[agent_name]['reward'], dtype=np.float32))
            dones.append(self._last_agent_to_done[agent_name])
            values.append(np.array(0., dtype=np.float32))
        advantages = self._pipeline.batch_process(agent_name, rewards, values, dones)

        # 算好advantage后将其append到fragments中
        # 这一块也比较难理解，advantage的长度还能和fragments不一样？
        # 先来看看advantage的计算，adv = r_t + gamma * v_(t+1) - v_t
        # 主要原因就是我们在算在s_t下执行a_t的advantage的时候，需要用到下一步v值。
        # 假设fragment的长度是n的话，我们能计算出的advantage数量只有n-1。
        fragments_len = len(fragments)
        advantages_len = len(advantages)

        for i in range(advantages_len):
            fragments[i].append({ADVANTAGE: np.asarray(advantages[i], dtype=np.float32)})
        if fragments_len == advantages_len + 1:
            fragments.pop(-1)
            # del self._dynamic_mask_history[agent_name]

        for i in reversed(range(len(fragments))):
            if drop_outs[i] == 1:
                fragments.pop(i)

    def render(self, **kwargs):
        if hasattr(self._env, 'render'):
            return self._env.render(**kwargs)    # type: ignore
        else:
            raise RuntimeError("self._env does not have a render method!")