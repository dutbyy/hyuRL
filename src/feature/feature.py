from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union
from typing_extensions import get_args

import numpy as np

Number = Union[int, float, np.integer, np.floating]
Array_Like = Union[List, np.ndarray]
Integer = Union[int, np.integer]
Floating = Union[float, np.floating]

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

class Feature(ABC):

    @abstractmethod
    def process(self, value) -> np.ndarray:
        """ 执行数据处理

        Parameters
        ----------
        value : Any
            需要被处理的数据

        Returns
        -------
        np.ndarray
            处理完的数据
        """
        raise NotImplementedError


class CellFeature(Feature):
    """
    如果说 FeatureSet 构成了超平面 (hyper-plane)，那么 Feature 为超平面上的一个 ***cell***。\
    FeatureSet 会沿着某一维度将超平面上的 cell 堆叠在一起，所以 Feature 的长度决定了堆叠维度的长度。\
    Drill 提供的所有 Feature 均基于该类开发。

    Parameters
    ----------
    length: int
        当前 Feature 在超平面上的堆叠维度上面的长度
    """

    def __init__(self, length: int) -> None:
        self._length = length
        self._expected_shape = None

    def process(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float32)
        return fast_to_ndarray(value)

    @property
    def length(self) -> int:
        """返回 Feature 在超平面上的堆叠维度上面的长度
        """
        return self._length

    def build_shape(self, shape: Tuple, index_value: Number) -> None:
        """为当前的 Feature 构建一个完整的，符合预期的 shape。需要注意的是 \
        这个 function 只应该被 FeatureSet 调用，而**不应该**由用户调用。

        Parameters
        ----------
        shape : Tuple
            预期的 shape
        index_value : Number
            用这个值去指代堆叠的维度，FeatureSet 将会沿此维度进行堆叠。

        Raises
        ------
        TypeError
            如果输入的 shape 不是 tuple, 则抛出 TypeError
        """
        if not isinstance(shape, Tuple):
            raise TypeError('shape must be tuple')

        self._expected_shape = tuple(
            value if value != index_value else self._length for value in shape)

    def copy(self) -> Feature:
        """生成一个关于当前 Feature 的复制，但是当前 Feature 的状态不会被复制。

        Returns
        -------
        Feature
            当前 Feature 的复制
        """
        return type(self)(self._length)

    def reset(self):
        """重置所有 runtime 相关的状态
        """


class PlainFeature(CellFeature):
    """
    PlainFeature 是用来处理一个标量值的 Feature，这个值必须是一个数字，而不能是一个 array。

    Examples
    --------
    >>> a = PlainFeature()
    >>> a.process(1)
    array([1.])
    """

    def __init__(self) -> None:
        super().__init__(1)

    def process(self, value: Number) -> np.ndarray:
        if not isinstance(value, get_args(Number)):
            raise TypeError(f"PlainFeature must receive a number, but got {type(value)}")
        # as plain feature will accept a literal, we must wrap it first
        return super().process([value])

    def copy(self):
        return PlainFeature()


class VectorFeature(CellFeature):
    """
    VectorFeature 用来处理一组数据，这组数据可以有任意的维度，但是必须是一个 array。

    Parameters
    ----------
    length : length
        当前 Feature 在超平面上的堆叠维度上面的长度

    Examples
    --------
    >>> a = VectorFeature(2)
    >>> a.process([1, 2])
    array([1., 2.])
    """

    def __init__(self, length: int):
        super().__init__(length)

    def process(self, value: Array_Like) -> np.ndarray:
        if not isinstance(value, get_args(Array_Like)):
            raise TypeError(f"VectorFeature must receive a list or array, but got {type(value)}")

        return super().process(value)


class RangedFeature(CellFeature):
    """
    RangedFeature 用来处理在一个特定的范围内的值。这个值可以是一个标量值，也可以是一个 array_like 的数据。


    Parameters
    ----------
    low : Number
        最小值
    high : Number
        最大值
    length : int, optional
        当前 Feature 在超平面上的堆叠维度上面的长度, 传入 1 代表处理的是标量。

    Examples
    --------
    >>> a = RangedFeature(0, 10)
    >>> a.process(5)
    array([0.5,])
    >>> a = RangedFeature(0, 10, 2)
    >>> a.process([5, 6])
    array([0.5, 0.6])
    """

    def __init__(self, low: Number, high: Number, length=1):
        self._low = low
        self._high = high

        assert low < high, 'low must be less than high'

        super().__init__(length)

    def process(self, value: Union[Number, Array_Like]) -> np.ndarray:
        if not isinstance(value, get_args(Union[Number, Array_Like])):
            raise TypeError(f"RangedFeature must receive a number or array, but got {type(value)}")

        if isinstance(value, get_args(Array_Like)):
            value = super().process(value)
        else:
            value = super().process([value])

        if any(value > self._high) or any(value < self._low):
            raise ValueError(
                f"RangedFeature must receive a value in range [{self._low}, {self._high}], but got {value}"
            )

        value = (value - self._low) / (self._high - self._low)
        return value

    def copy(self):
        return RangedFeature(self._low, self._high, self._length)


class OnehotFeature(CellFeature):
    """OnehotFeature 可以将一个整数转换为一个 one-hot 的表示向量。

    Parameters
    ----------
    depth : Integer
        One-hot 向量最大长度，表示一共有多少个类别。

    Examples
    --------
    >>> a = OnehotFeature(3)
    >>> a.process(1)
    array([0., 1., 0.])
    """

    def __init__(self, depth: Integer):
        assert depth > 1, 'depth must be greater than 1'
        super().__init__(depth)

    def process(self, value: Integer) -> np.ndarray:
        if not isinstance(value, get_args(Integer)):
            raise TypeError(f'OnehotFeature must receive an integer, but got {type(value)}')

        if value < 0 or value >= self._length:
            raise ValueError(
                f'OnehotFeature must receive a value in range [0, {self._length - 1}], but got {value}'
            )

        res = [0.0] * self._length
        res[value] = 1.0
        return super().process(res)


class BinaryhotFeature(CellFeature):
    """BinaryHotFeature 可以将一个整数转换为一个二进制的表示向量。

    Parameters
    ----------
    depth : Integer
        BinaryHot 的深度，用来控制当前 Feature 能表示的最大值，为 $2^{depth} - 1$

    Examples
    --------
    >>> a = BinaryhotFeature(3)
    >>> a.process(6)
    array([1., 1., 0.,])
    """

    def __init__(self, depth: Integer) -> None:
        assert depth > 1, 'depth must be greater than 1'
        super().__init__(depth)
        self._max_value = pow(2, depth) - 1

    def process(self, value: Integer) -> np.ndarray:
        if not isinstance(value, get_args(Integer)):
            raise TypeError(f'BinaryhotFeature must receive an int, but got {type(value)}')

        if value < 0 or value > self._max_value:
            raise ValueError(
                f"BinaryhotFeature must receive a value in range [0, {self._max_value}], but got {value}"
            )

        value = (value & (1 << np.arange(self._length))) > 0

        return super().process(value)


class RepeatFeature(CellFeature):
    """RepeatFeature 本身不进行数据处理，但是可以自动重复在 \
    内部创建多个 Feature，然后把这些 Feature 的结果合并起来。

    Parameters
    ----------
    max_length : int
        Feature 的最大长度，如果在处理的时候没有达到最大值，那么就会填充 0。\
        如果输入的长度超过了预定的最大长度，那么会抛出异常。
    sub_features: Dict[str, CellFeature]
        被用来重复创建和处理数据的 Feature.

    Examples
    --------
    >>> a = RepeatFeature(3, {'a': RangedFeature(0, 10), 'b': RangedFeature(0, 10)})
    >>> a.build_shape((-1,), -1)   # this will be called by the FeatureSet
    >>> a.process([{'a': 1, 'b': 2}, {'a': 3,'b': 4}])
    array([0.1, 0.2, 0.3, 0.4, 0., 0.])
    """

    def __init__(self, max_length: int, sub_features: Dict[str, CellFeature]) -> None:
        self._sub_features = {}
        sorted_key = sorted(sub_features.keys())
        for key in sorted_key:
            self._sub_features[key] = sub_features[key]

        self._max_length = max_length
        length = sum(feature.length for feature in sub_features.values()) * max_length
        super().__init__(length)

    def process(self, value: List[Dict[str, Any]]) -> np.ndarray:
        if not isinstance(value, get_args(Array_Like)):
            raise TypeError(f'RepeatFeature must receive a list, but got {type(value)}')

        if len(value) > self._max_length:
            raise ValueError(f'RepeatFeature must receive a list with \
                    length less than {self._max_length}, but got {len(value)}')

        res = []

        for item in value:
            if not isinstance(item, Dict):
                raise TypeError(f'RepeatFeature must receive a list of dict, \
                        but got {type(item)} in list')
            for name, value in item.items():
                # when process the sub-feature, we wrap the key with the name
                # of the RepeatFeature
                try:
                    res.append(self._sub_features[name].process(value))
                except Exception as e:
                    import sys
                    raise type(e)(f"[{name}] in RepeatFeature: {e}").with_traceback(
                        sys.exc_info()[2])
        res = np.concatenate(res, axis=self._stack_dim)
        if res.shape != self._expected_shape:
            # calculate the padding target
            pad_len = tuple(
                self._expected_shape[i] - res.shape[i] for i in range(len(self._expected_shape)))
            # (0, i) means pad the result with 0 after the value we have
            # in the n-th dimension to make it the same shape as expected
            res = np.pad(res, tuple((0, i) for i in pad_len), 'constant', constant_values=0)

        return res

    def build_shape(self, shape: Tuple, index_value: Number) -> None:
        self._stack_dim = shape.index(index_value)
        super().build_shape(shape, index_value)
        self._expected_shape = tuple(
            value if value != index_value else self._length for value in shape)

    def copy(self):
        return RepeatFeature(self._max_length, {k: v.copy() for k, v in self._sub_features.items()})

    def reset(self):
        for feature in self._sub_features.values():
            feature.reset()


class FeatureWrapper(Feature):
    """FeatureWrapper 可以用来包装一些特征来进行简单的 Feature 的扩展。主要被用于在 Feature \
    处理前对输入进行一些预处理。

    Parameters
    ----------
    pre_funcs : List[Callable]
        Functions that is going to proform pre-processing
    base : Feature
        The base feature that will be wrapped

    Examples
    --------
    >>> a = FeatureWrapper([lambda x: x + 1, lambda x: x * 2], RangedFeature(0, 10))
    >>> a.build_shape((-1,), -1)   # this will be called by the FeatureSet
    >>> a.process(1)
    array([0.4])
    """

    def __init__(self, pre_funcs: List[Callable], base: Feature):
        self._base = base
        self._pre_funcs = pre_funcs
        self._pre_check()

    def _pre_check(self):
        """Perform type checking when instantiating the wrapper
        """
        assert isinstance(self._base,
                          Feature), f'base must be a Feature, but got {type(self._base)}'
        for f in self._pre_funcs:
            assert callable(f), f'Each pre-process unit must be a callable, but got {type(f)}'

    def process(self, value: Any, *args, **kwargs) -> np.ndarray:
        """Process the feature

        Parameters
        ----------
        value : Any
            The input value for the pre-processing and underlying feature

        Returns
        -------
        np.ndarray
            A numpy array that contains the processed value.
        """
        for f in self._pre_funcs:
            value = f(value)
        return self._base.process(value, *args, **kwargs)

    def unwrap(self) -> Feature:
        """Return the base feature

        Returns
        -------
        Feature
            The base feature
        """
        return self._base

    def copy(self) -> Feature:
        """
        Creating a copy of the FeatureWrapper, notice that the underlying feature \
        will also be copied

        Returns
        -------
        FeatureWrapper
            A brand new FeatureWrapper that is exactly the same as this one
        """
        return FeatureWrapper(self._pre_funcs, self._base.copy())

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __getstate__(self):
        # define this to make the class picklable
        return vars(self)

    def __setstate__(self, state):
        # define this to make the class unpicklable
        vars(self).update(state)
