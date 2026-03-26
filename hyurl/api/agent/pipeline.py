from typing import Any
from abc import ABC, abstractmethod
from hyurl.api.agent_type import ObsData, ActionData


class PipelineInterface(ABC):
    @staticmethod
    @abstractmethod
    def o2s(obs_data: ObsData, history=None) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def a2c(action_data: ActionData, history=None) -> ActionData:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reward(obs_data: ObsData, history=None) -> float:
        raise NotImplementedError
