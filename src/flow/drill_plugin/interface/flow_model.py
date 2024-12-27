from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import logging
import numpy as np
import tree
import torch

from drill import summary
from drill.keys import ACTION_MASK, DECODER_MASK
from drill.model import Model
from drill.utils import get_hvd

from drill.builder import Builder
    
    
def getLogger(env_id):
    log_name = env_id if isinstance(env_id, str) else f"env-{env_id}"
    logger = logging.getLogger(f"env-{env_id}")
    logger.setLevel(20)
    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    try:
        import os
        os.system("mkdir -p /job/logs/user_log/")
        handler = logging.FileHandler(f"/job/logs/user_log/{log_name}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except:
        pass
    return logger

# 定义一个递归函数来处理嵌套结构
def trans2tensor(nested_structure):   
    try: 
        if torch.cuda.is_available():
            return tree.map_structure(lambda x: torch.from_numpy(x).cuda(), nested_structure)
        else:
            return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)
    except Exception as e:
        return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)
    
# 定义一个递归函数来处理嵌套结构
def trans2numpy(nested_structure):   
    return tree.map_structure(lambda x: x.cpu().numpy(), nested_structure)
    
class FlowModelPPOSync:

    def __init__(self, model_name: str, builder: Builder):
        self._init(model_name, builder)
        self.logger = getLogger(f"{model_name}-PPO")

    def __getstate__(self):
        return self._model_name, self._builder, self._model._network.state_dict()

    def setstate_learn(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)
        self._model._network.load_state_dict(weights)
        self._model._network.train()
        self.logger = getLogger(f"{model_name}-learn")
        self.logger.info("calling set state learn")
        if torch.cuda.is_available():
            self._model._network.cuda()

    def setstate_predict(self, state=None):
        if state:
            model_name, builder, weights = state
            self._init(model_name, builder)
            self._model._network.load_state_dict(weights)
        self._model._network.requires_grad_(False)
        self._model._network.eval()
        self.logger = getLogger(f"{self._model_name}-predict")
        self.logger.info("calling set state predict")

        if torch.cuda.is_available():
            self._model._network.cuda()

    def get_weights(self) -> List[np.ndarray]:
        self.logger.info("calling get_weights")
        return [p.cpu().detach().numpy() for p in self._model._network.parameters()]


    def set_weights(self, weights: List[np.ndarray]):
        self.logger.info("calling set_weights")
        with torch.no_grad():
            for target_p, p in zip(self._model._network.parameters(), weights):
                target_p.copy_(torch.from_numpy(p))

    def save_weights(self, mode='pth'):
        self.logger.info("calling save_weights")
        model_path = f'{self._save_params["path"]}/{self._model_name}/{self._model_name}_{self._learn_step}.pth'
        torch.save(self._model._network.state_dict(), model_path)


    def load_weights(self, model_path: str, backend: str='pytorch', mode='npz'):
        self.logger.info("calling load_weights")
        self._model._network.load_state_dict(torch.load(model_path))


    def _init(self, model_name, builder: Builder):
        self._model_name = model_name
        self._model: Model = builder.build_model(model_name)
        self._builder = builder
        self._learn_step = builder.learn_step
        self._update_step = 0
        if model_name in builder.save_params:
            self._save_params = builder.save_params[model_name]
            Path(f'{self._save_params["path"]}/{self._model_name}').mkdir(parents=True, exist_ok=True)
        self.logger = None
        self.last_learn = None


    def __init_logger(self):
        self.logger = logging.getLogger(f"model_learn")
        self.logger.setLevel(10)
        self._model.logger = self.logger
        if not self.logger.handlers:
            try:
                handler = logging.FileHandler(f"/job/logs/logs/model_learn.log")
            except:
                handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(name)s] [%(asctime)s] [%(filename)s:%(lineno)s] %(message)s'))
            self.logger.addHandler(handler)
        self.logger.info('Logger Init Finished')

    def learn(self, piece: List[Dict[str, Any]]) -> bool:
        if not self.logger:
            self.__init_logger()
        state_dict, behavior_info_dict, mask_dict, advantage = piece
        behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]
        
        training_data = {"state_dict": state_dict}
        training_data.update(behavior_info_dict)
 
        training_data = trans2tensor(training_data)
        summary_dict = self._model.learn(training_data)
        summary_dict= trans2numpy(summary_dict)

        self._learn_step += 1
        if True:
            summary.sum(f"{self._model_name}_update_step", 1, source="origin")
            if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
                self.save_weights(self._save_params["mode"])
                self.logger.info("saving weights of model.")
        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = trans2tensor(state_dict)
        predict_output_dict = self._model.predict(state_dict)
        output_dict = tree.map_structure(lambda x: x.cpu().detach().numpy(), predict_output_dict)
        return output_dict
