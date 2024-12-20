from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from drill.pipeline.agent_pipeline import AgentPipeline
from drill.pipeline.global_pipeline import GlobalPipeline
from drill.pipeline.interface import ActionData, History, ObsData
from drill.utils import construct


class PipelineManager:

    def __init__(self,
                 agent_pipeline_info: Dict[str, Dict[str, Any]],
                 global_pipeline: GlobalPipeline,
                 history_len=0) -> None:
        self._history = History(history_len)
        self._ap_info = agent_pipeline_info
        self._history_len = history_len
        self._agent_to_pipeline: Dict[str, AgentPipeline] = {}
        self._global_pipeline = global_pipeline

        self._post_cache: Dict[str, ActionData] = {}
        self._obs_count = 0

    def pre_process(self, data: Any, episode_done: bool) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """Perform pre-processing on the input data for each given agent.

        Returns
        -------
        Tuple[Dict, Dict]
            The returned value
        """
        global_processed_dict, self._history = self._global_pipeline.pre_process(
            data, self._history)
        if not isinstance(global_processed_dict, Dict):
            raise ValueError('Value returned by global pipeline must be a dict')

        self._obs_count = len(global_processed_dict)
        res_state = {}
        res_reward = {}

        # perform the local processing
        for agent_name, info in global_processed_dict.items():
            # create new local pipeline if new agent is encountered
            if agent_name not in self._agent_to_pipeline:
                self._init_new_agent(agent_name)

            try:
                if not isinstance(info, ObsData):
                    raise ValueError(
                        f'Value returned by global pipeline must be a ObsData dataclass')
                info.agent_name = agent_name
                info.extra_info_dict["episode_done"] = episode_done
                # TODO: how to use the monitor return?
                res_state[agent_name], res_reward[agent_name] = self._agent_to_pipeline[
                    agent_name].pre_process(info, self._history)
            except Exception as e:
                import sys
                raise type(e)(f'Failed when processing [{agent_name}] due to: {e}').with_traceback(
                    sys.exc_info()[2])

        return res_state, res_reward

    def post_process(
        self, agent_name: str, data: ActionData
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]], Dict[Any, np.ndarray]]:
        """Perform post-process for users.

        Parameters
        ----------
        agent_name : str
            The name of the agent that the current data belongs to
        data : ActionData
            The data to be post-processed

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Any, Dict[Any, np.ndarray]]
            The return contains action_dict, action_mask, decoder_mask

        Raises
        ------
        ValueError
            If the return of the agentPipeline is not a ActionData, we will raise a ValueError
        """

        received_actions = list(data.action.keys())

        # the return should contains actions that will be chosen for each head, and logits that
        # each action was sampled from. If the value of
        # any head is not necessary, simply do not return it. We will mask for the user.
        data.agent_name = agent_name
        post_result = self._agent_to_pipeline[agent_name].post_process(data, self._history)

        if not isinstance(post_result, ActionData):
            raise ValueError(f'Value returned by post-process must be a ActionData dataclass,\
                    but have {type(post_result)}')

        # construct the decoder mask from the user returned action dict
        decoder_mask = {
            action_name: np.array(action_name in post_result.action, dtype=np.float32)
            for action_name in received_actions
        }

        # cache all the actions and only return if we collect enough
        self._post_cache[agent_name] = post_result
        if len(self._post_cache) == self._obs_count:
            per_agent_dict, self._history = self._global_pipeline.post_process(
                self._post_cache, self._history)
            # reset the record
            self._post_cache = {}
            self._obs_count = 0

            action_mask = {
                agent_name: per_agent_dict[agent_name].action_mask for agent_name in per_agent_dict
            }
            action_dict = {
                agent_name: per_agent_dict[agent_name].action for agent_name in per_agent_dict
            }
            return action_dict, action_mask, decoder_mask

        return None, {}, decoder_mask

    def batch_process(self, agent_name: str, rewards: List, values: List, dones: List) -> List:
        return self._agent_to_pipeline[agent_name].batch_process(rewards, values, dones)

    def get_fake_state(self, agent_name: str, batch_size: int = 0) -> Dict[str, np.ndarray]:
        if agent_name not in self._agent_to_pipeline:
            self._init_new_agent(agent_name)

        if hasattr(self._agent_to_pipeline[agent_name].feature_handler, 'get_fake_state'):
            return self._agent_to_pipeline[agent_name].feature_handler.get_fake_state(batch_size)
        else:
            raise ValueError('Feature Handler does not have get_fake_state class method')

    def _init_new_agent(self, agent_name: str):
        if agent_name not in self._ap_info:
            raise ValueError(f'{agent_name} is not a reconginzed agent name')
        self._agent_to_pipeline[agent_name] = construct(self._ap_info[agent_name])

    def reset(self):
        """ 通常，会在每个 episode 结束后调用 reset。
        注意：这里没有调用 history.clear()，drill 希望保留不同 episode 之前的 history。
        当然用户可以选择手动清空 history。
        """
        self._post_cache = {}
        self._obs_count = 0

        self._global_pipeline.reset()
        for ap in self._agent_to_pipeline.values():
            ap.reset()
