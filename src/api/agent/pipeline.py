
from typing import List, Dict, Any, OrderedDict, Union
from abc import ABC, abstractmethod
from api.agent_type import ObsData, ActionData


class PipelineInterface(ABC):
    @staticmethod
    @abstractmethod
    def o2s(obs_data: ObsData):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def a2c(action_data: ActionData):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reward(obs_data: ObsData):
        raise NotImplementedError
