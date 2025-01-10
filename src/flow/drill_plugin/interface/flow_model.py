from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from drill.flow import flow

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
    if torch.cuda.is_available():
        return nested_structure
    else:
        return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)

# 定义一个递归函数来处理嵌套结构
def trans2numpy(nested_structure):
    return tree.map_structure(lambda x: x.cpu().numpy(), nested_structure)

class FlowModelPPO(flow.Model):

    def __init__(self, model_name: str, builder: Builder):
        self._init(model_name, builder)
        self.logger = getLogger(f"{model_name}-master")

    def __getstate__(self):
        self.logger.info("calling __getstate__")
        return self._model_name, self._builder, self._model._network.state_dict()

    def setstate_learn(self, state):
        try:
            model_name, builder, weights = state
            if not hasattr(self, "logger"):
                self.logger = getLogger(f"{model_name}-learner")
            self.logger.info(f"model_name : {model_name}")
            self.logger.info(f"builder : {builder}")
            weight_shape = {k: v.shape for k, v in weights.items()}
            self.logger.info(f"weights : {weight_shape}")
            self._init(model_name, builder)
            self._model._network.load_state_dict(weights)
        except Exception as e:
            self.logger.info(f"setstate learn error: {e}")
            raise e
        self._model._network.train()
        if torch.cuda.is_available():
            self._model._network.cuda()

    def setstate_predict(self, state=None):
        try:
            model_name, builder, weights = state
            if not hasattr(self, "logger"):
                self.logger = getLogger(f"{model_name}-predictor")
            self.logger.info("calling set state predict")
            self.logger.info(f"model_name : {model_name}")
            self.logger.info(f"builder : {builder}")
            weight_shape = {k: v.shape for k, v in weights.items()}
            self.logger.info(f"weights : {weight_shape}")
            self._init(model_name, builder)
            self._model._network.load_state_dict(weights)
        except Exception as e:
            self.logger.info(f"setstate predict error: {e}")
            raise e

        self._model._network.requires_grad_(False)
        self._model._network.eval()

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
        self.last_learn = None


    def learn(self, piece: List[Dict[str, Any]]) -> bool:
        state_dict, behavior_info_dict, mask_dict, advantage = piece
        behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]

        training_data = {"state_dict": state_dict}
        training_data.update(behavior_info_dict)

        training_data = trans2tensor(training_data)
        summary_dict = self._model.learn(training_data)
        summary_dict= trans2numpy(summary_dict)

        if True:
            summary.sum(f"{self._model_name}_update_step", 1, source="origin")
            if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
                self.save_weights(self._save_params["mode"])
                self.logger.info("saving weights of model.")

        self._learn_step += 1
        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = trans2tensor(state_dict)
        predict_output_dict = self._model.predict(state_dict)
        output_dict = tree.map_structure(lambda x: x.cpu().detach().numpy(), predict_output_dict)
        return output_dict
