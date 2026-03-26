from hyurl.api.agent_type import ActionData, ObsData
from .agent_pipeline import Agent
from typing import Tuple, Any
from rich import print


class PipelineManager:
    def __init__(self, agent_piplines: dict[str, dict]):
        self._agent_piplines = agent_piplines
        self.history = None

    def reset(self):
        pass

    def pre_process(
        self, obs_data_dict: dict[str, ObsData], episode_done: bool
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        预处理 obs_data_dict
        """
        return {
            k: self._agent_piplines[k].o2s(v, self.history)
            for k, v in obs_data_dict.items()
        }, {
            k: self._agent_piplines[k].reward(v, self.history)
            for k, v in obs_data_dict.items()
        }

    def post_process(self, agent_name: str, action_data: ActionData):
        import numpy as np
        return {agent_name: action_data.action}, action_data.action_mask, {k: np.array(1) for k in action_data.action}
