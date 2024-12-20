from drill.pipeline.interface import ActionData, ObsData

class PipelineImplement:
    @staticmethod
    def feature_handler(obs_data:ObsData, history):
        return {
            "common": {
                "raw": obs_data.obs
            }
        }
   
    @staticmethod
    def reward_handler(obs_data:ObsData, history):
        return 1
         
    @staticmethod
    def action_handler(action_data:ActionData, history):
        return action_data

    