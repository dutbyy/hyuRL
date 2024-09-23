from typing import Dict
import logging


def is_class_dict(class_dict: Dict):
    return ("class" in class_dict) and ("params" in class_dict)


def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件"""

    if not is_class_dict(class_dict):
        raise ValueError(
            f"Expected a dict with keys 'class' and 'params', but got {class_dict}"
        )

    class_ = class_dict["class"]
    params = class_dict["params"]
    # print(class_, params)
    return class_(**params)


def timer_decorator(func):
    import timeit

    def wrapper(*args, **kwargs):
        start = timeit.default_timer() * 1000
        result = func(*args, **kwargs)
        end = timeit.default_timer() * 1000
        eplased_time = round(end - start, 1)
        logger = logging.getLogger(f"model_learn")
        # logger.info(f"函数 {func.__name__} 执行时间: {eplased_time} ms")
        return result

    return wrapper


# 实现一个单例模式
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def reset(self):
        self.infer_times = []

    def add_infer_time(self, eplased_time):
        self.infer_times.append(eplased_time)

    def print_info(self):
        print(f"Infer times: {self.infer_times}")
