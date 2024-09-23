from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import drill
import numpy as np
import tree

from drill import summary
from drill.flow import flow
from drill.keys import ACTION_MASK, DECODER_MASK
from drill.model import Model
from drill.utils import get_hvd

if TYPE_CHECKING:
    from drill.builder import Builder


class FlowModelPPO(flow.Model):
    """FlowModel 是 [flow.api.Model](https://docs.inspir.work/flow/flow/tutorial.html#flow.api.Model) 一个实现。

    Attributes
    ----------
    model_name : str
        model 的名字
    builder: Builder
        详见 `drill.builder.Builder`
    """

    def __init__(self, model_name: str, builder: Builder):
        self._init(model_name, builder)

    def _init(self, model_name, builder: Builder):
        self._model_name = model_name
        self._model: Model = builder.build_model(model_name)
        self._builder = builder
        self._sync = self._builder._models[model_name]['params'].get('sync', False)
        self._sync_interval = self._builder._models[model_name]['params'].get('sync_interval', 10)
        self._learn_step = builder.learn_step
        self._update_step = 0
        hvd = get_hvd(builder.backend)
        if hvd.rank() == 0 and (model_name in builder.save_params):
            self._save_params = builder.save_params[model_name]
            Path(f'{self._save_params["path"]}/{self._model_name}').mkdir(parents=True, exist_ok=True)

    @property
    @functools.lru_cache()
    def _is_tensorflow(self):
        return self._builder.backend == "tensorflow"

    def get_weights(self) -> List[np.ndarray]:
        """ 获取模型权重

        Returns
        -------
        List[np.ndarray]
            模型权重
        """
        if self._is_tensorflow:
            return self._model.network.get_weights()
        return [p.cpu().detach().numpy() for p in self._model.network.parameters()]

    def set_weights(self, weights: List[np.ndarray]):
        """ 设置模型权重

        Parameters
        ----------
        weights : List[np.ndarray]
            模型权重
        """
        if self._is_tensorflow:
            self._model.network.set_weights(weights)
        else:
            import torch
            for target_p, p in zip(self._model.network.parameters(), weights):
                target_p.copy_(torch.from_numpy(p))

    def save_weights(self, mode='npz'):
        """ 保存模型

        Parameters
        ----------
        mode : str, optional
            模型的格式，可以是 'npz' 或者 'tf-ckpt'， by default 'npz'
        """
        from drill.utils import save_model
        model_path = f'{self._save_params["path"]}/{self._model_name}/{self._model_name}_{self._learn_step}'
        save_model(self._model.network, model_path, self._builder.backend, mode)

    def load_weights(self, model_path: str, backend: str, mode='npz'):
        """ 加载模型

        Parameters
        ----------
        model_path : str
            模型路径，注意路径应指向具体模型文件，而不是其父级目录
        mode : str, optional
            模型的格式，可以是 'npz' 或者 'tf-ckpt'， 默认为 'npz'
        """
        from drill.utils import load_model
        load_model(self._model.network, model_path, backend, mode)

    def __getstate__(self):
        if self._is_tensorflow:
            return self._model_name, self._builder, self.get_weights()
        return self._model_name, self._builder, self._model.network.state_dict()

    def setstate_learn(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)

        if self._is_tensorflow:
            self.set_weights(weights)
        else:
            import torch
            self._model.network.load_state_dict(weights)
            if torch.cuda.is_available():
                self._model.network.cuda()

    def setstate_predict(self, state):
        model_name, builder, weights = state
        self._init(model_name, builder)

        if self._is_tensorflow:
            self.set_weights(weights)
        else:
            import torch
            self._model.network.load_state_dict(weights)
            self._model.network.requires_grad_(False)
            self._model.network.eval()
            if torch.cuda.is_available():
                self._model.network.cuda()

    def learn(self, piece: List[Dict[str, Any]]) -> bool:
        """ `FlowModel` 使用批量数据 piece 进行学习，训练模型

        Parameters
        ----------
        piece : List[Dict[str, Any]]
            由 state_dict, behavior_info_dict, decoder_mask, advantage 组成。

            * state_dict 包含 state， reward， done 等信息，还可能包含 hidden_state;
            * behavior_info_dict 包含 logits, action, value;
            * decoder_mask 包含 valid action。

        Returns
        -------
        bool
            是否将数据推送给 PredictorService
        """
        if hasattr(self, "_save_params") and self._learn_step % self._save_params["interval"] == 0:
            self.save_weights(self._save_params["mode"])
        self._learn_step += 1

        hvd = get_hvd(self._builder.backend)
        if hvd.rank() == 0:
            summary.sum(f"{self._model_name}_learn_step", 1, source="origin")
        state_dict, behavior_info_dict, mask_dict, advantage = piece

        behavior_info_dict.update(advantage)
        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]

        # for name, mask in mask_dict[ACTION_MASK].items():
        #     if name not in state_dict:
        #         state_dict[name] = mask
        #     elif len(state_dict[name]) != len(mask):
        #         raise ValueError(f"{name} mask length not equal, there might be a name duplication")
        #     else:
        #         state_dict[name] = np.asarray(state_dict[name], dtype=np.float32) & \
        #                                     np.asarray(mask, dtype=np.float32)

        # summary 包含 loss, entropy, policy_loss, value_loss
        summary_dict = self._model.learn(state_dict, behavior_info_dict)

        for k, v in summary_dict.items():
            summary.average(k, v)
        if (self._sync and self._learn_step % self._sync_interval == 0) or not self._sync:
            if hvd.rank() == 0:
                summary.sum(f"{self._model_name}_update_step", 1, source="origin")
            return True
        else:
            return False

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
        predict_output_dict = self._model.predict(state_dict)
        output_dict = tree.map_structure(lambda x: x.numpy(), predict_output_dict)
        return output_dict