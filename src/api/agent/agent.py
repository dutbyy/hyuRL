
from typing import List, Dict, Any, OrderedDict, Union
from api.agent_type import ObsData, ActionData
from feature.feature_set import FeatureSet


class Agent:
    def __init__(self, feature_sets: List[FeatureSet]):
        self.feature_sets = feature_sets
    
    def o2s(self, obs_data: ObsData):
        raise NotImplementedError
    
    def a2c(self, action_data: ActionData):
        raise NotImplementedError
    
    def reward(self, obs_data: ObsData):
        raise NotImplementedError


    def feature_process(self, feature_dict: Dict[str, Union[Dict, List]]):
        input_dict = OrderedDict()
        for feature_set in self.feature_sets:
            input_dict[feature_set.name] = feature_set.process(feature_dict[feature_set.name])