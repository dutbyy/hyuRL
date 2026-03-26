import numpy as np
from hyurl.api.agent_type import ObsData, ActionData


class PipelineImplement:
    @staticmethod
    def o2s(obs_data: ObsData, history):
        return {
            "common": {
                "raw": obs_data.obs
            }
        }

    @staticmethod
    def reward(obs_data: ObsData, history):
        return obs_data.extra_info_dict.get("reward", 1.0)

    @staticmethod
    def a2c(action_data: ActionData, history):
        action_data.action_mask = {k: np.ones(1) for k in action_data.action}
        return action_data
