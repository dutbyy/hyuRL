from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import logging
import traceback
import numpy as np
import tree
import torch

from drill import summary
from drill.flow import flow
from drill.keys import ACTION_MASK, DECODER_MASK
from drill.model import Model
from drill.utils import get_hvd

from drill.builder import Builder
    
    
# 定义一个递归函数来处理嵌套结构
def trans2tensor(nested_structure):   
    try: 
        if torch.cuda.is_available():
            return tree.map_structure(lambda x: torch.from_numpy(x.numpy()).cuda(), nested_structure)
        else:
            return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)
    except:
        return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)
    
# 定义一个递归函数来处理嵌套结构
def trans2numpy(nested_structure):   
    return tree.map_structure(lambda x: x.cpu().numpy(), nested_structure)
    
class FlowModelPPOSync(flow.Model):

    def __init__(self, model_name: str, builder: Builder):
        self._init(model_name, builder)

    def __getstate__(self):
        return self._model_name, self._builder, self._model._network.state_dict()

    def setstate_learn(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)
        self._model._network.load_state_dict(weights)

        if torch.cuda.is_available():
            self._model._network.cuda()

    def setstate_predict(self, state=None):
        if state:
            model_name, builder, weights = state
            self._init(model_name, builder)
            self._model._network.load_state_dict(weights)
        self._model._network.requires_grad_(False)
        self._model._network.eval()

        if torch.cuda.is_available():
            self._model._network.cuda()

    def get_weights(self) -> List[np.ndarray]:
        return [p.cpu().detach().numpy() for p in self._model._network.parameters()]

    def set_weights(self, weights: List[np.ndarray]):
        for target_p, p in zip(self._model._network.parameters(), weights):
            target_p.copy_(torch.from_numpy(p))

    def save_weights(self, mode='npz'):
        from drill.utils import save_model
        model_path = f'{self._save_params["path"]}/{self._model_name}/{self._model_name}_{self._learn_step}'
        save_model(self._model._network, model_path, self._builder.backend, mode)

    def load_weights(self, model_path: str, backend: str, mode='npz'):
        from drill.utils import load_model
        load_model(self._model._network, model_path, backend, mode)


    def _init(self, model_name, builder: Builder):
        self._model_name = model_name
        self._model: Model = builder.build_model(model_name)
        self._builder = builder
        self._sync = self._builder._models[model_name]['params'].get('sync', False)
        self._sync_interval = self._builder._models[model_name]['params'].get('sync_interval', 1)
        self._learn_step = builder.learn_step
        self._update_step = 0
        # hvd = get_hvd(builder.backend)
        # if hvd.rank() == 0 and (model_name in builder.save_params):
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

        self.logger.info('begin to calc model learn')

        state_dict, behavior_info_dict, mask_dict, advantage = piece
        behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]

        traning_data = {"state_dict": state_dict}
        traning_data.update(behavior_info_dict)

        traning_data = trans2tensor(traning_data)


        try:
            summary_dict = self._model.learn(traning_data)
        except Exception as e:
            exc_info = traceback.format_exception(type(e), e, e.__traceback__)
            exc_message = "".join(exc_info)
            raise ValueError(f"origin error : {e}\n learn_message : {exc_message}")

        summary_dict= trans2numpy(summary_dict)

        try:
            for k, v in summary_dict.items():
                summary.average(k, v)
        except Exception as e:
            exc_info = traceback.format_exception(type(e), e, e.__traceback__)
            exc_message = "".join(exc_info)
            raise ValueError(f"origin error : {e}\n learn_message : {exc_message} \n{summary_dict}")
        
        
        # hvd = get_hvd(self._builder.backend)
        self._learn_step += self._sync_interval
        # if hvd.rank() == 0:
        if True:
            summary.sum(f"{self._model_name}_learn_step", self._sync_interval, source="origin")
            summary.sum(f"{self._model_name}_update_step", 1, source="origin")
            if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
                self.save_weights(self._save_params["mode"])
        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = trans2tensor(state_dict)
        predict_output_dict = self._model.predict(state_dict)
        output_dict = tree.map_structure(lambda x: x.cpu().detach().numpy(), predict_output_dict)
        return output_dict
