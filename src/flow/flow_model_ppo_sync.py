from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import drill
import numpy as np
import tree
import torch
import tensorflow as tf

from drill import summary
from drill.flow import flow
from drill.keys import ACTION_MASK, DECODER_MASK
from drill.model import Model
from drill.utils import get_hvd
import logging
import timeit
import traceback
if TYPE_CHECKING:
    from drill.builder import Builder


class Clocker:
    def __init__(self):
        self.time_lis = []
    
    def tick(self):
        self.time_lis.append(timeit.default_timer() * 1000)
    
    def show(self, logger):
        show_str = ', '.join( [f'{round(j-i, 1)} ms' for i, j in zip(self.time_lis[:-1], self.time_lis[1:])])
        logger.info(f"eplased times: {show_str}")
    
# 定义一个递归函数来处理嵌套结构
def trans2tensor(nested_structure):
    if torch.cuda.is_available():
        return tree.map_structure(lambda x: torch.from_numpy(x.numpy()).cuda(), nested_structure)
    else:
        return tree.map_structure(lambda x: torch.from_numpy(x), nested_structure)

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

    def setstate_predict(self, state):
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
        hvd = get_hvd(builder.backend)
        if hvd.rank() == 0 and (model_name in builder.save_params):
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

        if self.last_learn:
            sample_batch_time = round( (time.time() - self.last_learn) * 1000, 1)
            # self.logger.info(f"model.learn 调用间隔 : {eplased_time} ms")
        else :
            sample_batch_time = -1

        clocker = Clocker()
        clocker.tick()
#         total_time = time.default_timer() * 1000
        self.logger.info('begin to calc model learn')

        state_dict, behavior_info_dict, mask_dict, advantage = piece
        behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]


#         import tensorflow as tf
#         state_dict = tree.map_structure(lambda x: tf.convert_to_tensor(x), state_dict)
#         behavior_info_dict = tree.map_structure(lambda x: tf.convert_to_tensor(x), behavior_info_dict)
#         trans_time = time.time() * 1000
#         clocker.tick()

        try :
            state_dict = trans2tensor(state_dict)
            behavior_info_dict = trans2tensor(behavior_info_dict)
        except Exception as e:
            exc_info = traceback.format_exception(type(e), e, e.__traceback__)
            exc_message = "".join(exc_info)
            raise ValueError(f"origin error : {e}\n learn_message : {exc_message}")
#         trans_eplased_time = round(time.time()*1000 - trans_time, 1)
        clocker.tick()


#         learn_time = time.time() * 1000
        try:
            summary_dict = self._model.learn(state_dict, behavior_info_dict, episode=self._sync_interval)
        except Exception as e:
            exc_info = traceback.format_exception(type(e), e, e.__traceback__)
            exc_message = "".join(exc_info)
            raise ValueError(f"origin error : {e}\n learn_message : {exc_message}")
#         learn_eplased_time = round(time.time()*1000 - learn_time, 1)
        clocker.tick()

        for k, v in summary_dict.items():
            summary.average(k, v)

        hvd = get_hvd(self._builder.backend)
        self._learn_step += self._sync_interval
        if hvd.rank() == 0:
            summary.sum(f"{self._model_name}_learn_step", self._sync_interval, source="origin")
            summary.sum(f"{self._model_name}_update_step", 1, source="origin")
            if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
                self.save_weights(self._save_params["mode"])
        clocker.tick()
        clocker.show(self.logger)

#         summary_eplased_time = round(time.time()*1000 - summary_time, 1)
#         total_eplased_time = round(time.time()*1000 - total_time, 1)


#         self.logger.info(f"调用learn间隔 [{sample_batch_time}ms] 调用trans耗时 [{trans_eplased_time}] 调用learn耗时 [{learn_eplased_time}] 调用summary耗时 [{summary_eplased_time}]")
#         self.last_learn = time.time()
        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ `FlowModel` 进行前向预测

        Parameters
        ----------
        state_dict : Dict[str, Any]
            模型 inference 的输入

        Returns
        -------
        Dict[str, Any]
            模型 inference 的结果

        Examples
        --------
        state_dict
        ``` python
        {
            "spatial": np.ndarray,
            "entity": np.ndarray,
            "reward": array,
            "hidden_state": Any,
            ...
        }
        ```

        return
        ```python
        {
            "logits": {
                "x": np.ndarray,
                "y": np.ndarray
            },
            "action": {
                "x": np.ndarray,
                "y": np.ndarray
            }
            "value": np.ndarray,
            "hidden_state": np.ndarray
        }
        ```
        """
#         state_dict = tree.map_structure(lambda x: tf.convert_to_tensor(x), state_dict)
        # behavior_info_dict = tree.map_structure(lambda x: tf.convert_to_tensor(x).cuda(), behavior_info_dict)
        state_dict = trans2tensor(state_dict)
        predict_output_dict = self._model.predict(state_dict)
        output_dict = tree.map_structure(lambda x: x.cpu().numpy(), predict_output_dict)
        return output_dict
