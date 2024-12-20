from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .feature import Feature

def fast_to_ndarray(value: List, dtype=np.float32) -> np.ndarray:
    from itertools import chain
    # fastest nested list to ndarray convertion
    if not isinstance(value[0], (List, np.ndarray)):
        # 1D array
        return np.fromiter(value, dtype)
    elif not isinstance(value[0][0], (List, np.ndarray)):
        # 2D array
        return np.fromiter(chain.from_iterable(value), dtype).reshape((len(value), -1))
    else:
        return np.asarray(value, dtype=dtype)
INDEX_VALUE = -1


class FeatureSet:
    """一个通用的基类，用于抽象 FeatureSet 的通用功能和接口。用户可以继承该类，实现自己的特征类型 \
    但是<u>**不应该**</u>实例化这个类。

    Parameters
    ----------
    feature_dict: Dictionary[str, Feature]
        一个包含所有属于当前 FeatureSet 的 Feature 的字典。
    shape: Tuple
        由 FeatureSet 构成的超平面的形状，对的 stacking dimension 应该与 INDEX_VALUE 对应。\
        比如，(2, 3, INDEX_VALUE) 表示所有 Feature 有 3 维，输出将在最后一个维度进行堆叠。\
        请注意，输入的 shape 需要有且仅有一个 stacking dimension.
    """

    def __init__(self, name: str, feature_dict: Dict[str, Feature], shape: Tuple):
        self._name = name
        self._feature_dict = {}
        sorted_key = sorted(feature_dict.keys())
        for key in sorted_key:
            self._feature_dict[key] = feature_dict[key]
        self._length = 0

        try:
            # find the stacking dimension
            self._stack_dim = shape.index(INDEX_VALUE)
        except:
            raise ValueError(f'The shape of {name} must have one \
                            INDEX_VALUE={INDEX_VALUE}, but got {shape}')

        for feature in self._feature_dict.values():
            self._length += feature.length
            feature.build_shape(shape, index_value=INDEX_VALUE)

        self._shape = tuple(value if value != -1 else self._length for value in shape)

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    def process(self, raw_value_dict: Dict[str, Any]) -> np.ndarray:
        """使用 feature_dict 对所有对应的输入进行处理

        Parameters
        ----------
        raw_value_dict : Dict[str, Any]
            所有需要被处理的输入，每个 key 对应一个 Feature

        Returns
        -------
        np.ndarray
            处理好的，且被 FeatureSet 聚合了的输出
        """
        output = self.process_feature(raw_value_dict)
        output = np.concatenate(output, axis=self._stack_dim)
        return output

    def process_feature(self, raw_value_dict: Dict[str, Any]) -> List[np.ndarray]:
        """使用 feature_dict 对 raw_value_dict 进行处理

        Parameters
        ----------
        raw_value_dict : Dict[str, Any]
            环境返回的原始 observation 的值

        Returns
        -------
        List[np.ndarray]
            处理后的 state
        """
        if not isinstance(raw_value_dict, dict):
            raise TypeError(f"raw_value_dict should be dict, but received {type(raw_value_dict)}")
        outputs = []

        for key in self._feature_dict:
            feature_processor = self._feature_dict[key]
            try:
                res = feature_processor.process(raw_value_dict[key])
                if res.shape != feature_processor._expected_shape:
                    raise ValueError(
                        f'The shape mismatch, expected {feature_processor._expected_shape}, got {res.shape}'
                    )
                outputs.append(res)
            except Exception as e:
                import sys
                raise type(e)(f"{self.name}[{key}]: {e}").with_traceback(sys.exc_info()[2])
        return outputs

    @property
    def length(self):
        return self._length

    def __str__(self) -> str:
        return self.name

    def reset(self):
        for f in self._feature_dict.values():
            f.reset()


class CommonFeatureSet(FeatureSet):
    """CommonFeatureSet 通常用于处理扁平的 Feature (只有一个维度)。

    Parameters
    ----------
    name: str
        FeatureSet 的名字，注意，这个名字是 FeatureSet 的重要标识符，请勿重复。
    feature_dict: Dict[str, Feature]
        一个包含所有属于当前 FeatureSet 的 Feature 的字典。
    """

    def __init__(self, name: str, feature_dict: Dict[str, Feature]):
        super().__init__(name, feature_dict, shape=(INDEX_VALUE,))


class EntityFeatureSet(FeatureSet):
    """EntityFeatureSet 通常用于被处理一种个体的合集，每一个合集中的个体具有的性质类似。比如，Dota2 \
       中的小兵，每个小兵都可以被表示为一个个体。与其创建一个 CommonFeatureSet 来表示每个小兵，\
       不如创建一个 EntityFeatureSet 来表示所有小兵，这样可以自动化的处理很多情况，比如小兵数量的增减。\
       需要注意的是，目前 EntityFeatureSet 也仅支持扁平的 Feature 结构 (每个 Feature 的维度只有一个)。

    Parameters
    ----------
    name: str
        FeatureSet 的名字，注意，这个名字是 FeatureSet 的重要标识符，请勿重复。
    feature_dict: Dict[str, Feature]
        一个包含所有属于当前 FeatureSet 的 Feature 的字典。
    max_length: int
        EntityFeatureSet 的 最大长度。处理后的特征会返回一个固定长度的数组，\
        如果输入的长度不足，则会在末尾补 0。
    
    Raises
    ------
    ValueError
        如果输入的数据不是一个 list 或者 array，或者输入长度大于 max_length，则抛出异常。
    """

    def __init__(self, name: str, feature_dict: Dict[str, Feature], max_length: int):
        super().__init__(name, feature_dict, shape=(INDEX_VALUE,))
        self._max_length = max_length

    @property
    def shape(self):
        return (self._max_length, self._length)

    def process(self, raw_values: List[Dict[str, Any]]) -> np.ndarray:
        if not isinstance(raw_values, List):
            raise ValueError(f'The process object for EntityFeatureSet \
                 {self._name} must be a list')

        if len(raw_values) > self._max_length:
            raise ValueError(f'The length of raw_values given to EntityFeatureSet \
                {self._name} for processing must be less than {self._max_length}')

        outputs = []
        for raw_value_dict in raw_values:
            output = super().process(raw_value_dict)
            outputs.append(output)

        # pad to fix length
        if len(outputs) < self._max_length:
            paddings = [
                [0. for _ in range(self._length)] for _ in range(self._max_length - len(outputs))
            ]
            outputs.extend(paddings)
        # return np.asarray(outputs, dtype=np.float32)
        return fast_to_ndarray(outputs)


class SpatialFeatureSet(FeatureSet):
    """SpatialFeatureSet 通常被用于处理高维数据，比如图像。

    Parameters
    ----------
    name: str
        FeatureSet 的名字，注意，这个名字是 FeatureSet 的重要标识符，请勿重复。
    feature_dict: Dict[str, Feature]
        一个包含所有属于当前 FeatureSet 的 Feature 的字典。
    shape : Tuple
        除了 channel 的其他 shape 信息，channel 维度会被自动处理
    channel : str, optional
        提供一个可选的 channel 参数，用于指定 channel 的位置，默认为 channel_last

    Raises
    ------
    ValueError
        如果 channel 不是 channel_first 或 channel_last 则会抛出异常
    """

    def __init__(self,
                 name: str,
                 feature_dict: Dict[str, Feature],
                 shape: Tuple,
                 channel='channel_last'):

        if channel == 'channel_first':
            shape = (INDEX_VALUE, *shape)
        elif channel == 'channel_last':
            shape = (*shape, INDEX_VALUE)
        else:
            raise ValueError(f"The channel must be 'channel_first' or \
                             'channel_last', but got {channel}")

        super().__init__(name, feature_dict, shape)
