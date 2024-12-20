
from ast import Not
from typing import List, Dict, Any, OrderedDict, Union
from api.agent_type import ObsData, ActionData
from feature.feature_set import FeatureSet
from .pipeline import PipelineInterface

class Agent:
    def __init__(self, feature_sets: List[FeatureSet], pipeline: PipelineInterface):
        self.__feature_sets = feature_sets
        self.__pipeline = PipelineInterface

    def o2s(self, obs_data: ObsData):
        return self.__pipeline.o2s(obs_data)
    
    def a2c(self, action_data: ActionData):
        return self.__pipeline.a2c(action_data)
    
    def reward(self, obs_data: ObsData):
        return self.__pipeline.reward(obs_data)

    def feature_process(self, feature_dict: Dict[str, Union[Dict, List]]):
        input_dict = OrderedDict()
        for feature_set in self.__feature_sets:
            input_dict[feature_set.name] = feature_set.process(feature_dict[feature_set.name])
    