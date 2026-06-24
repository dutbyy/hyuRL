from collections import defaultdict
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import os
import rich
from rich import pretty

def is_class_dict(class_dict: Dict):
    return ("class" in class_dict) and ("params" in class_dict)


def construct(class_dict: Dict):
    """根据 config dict, 从对应的 network component class 中实例化一个对应的网络组件"""

    if not is_class_dict(class_dict):
        raise ValueError(
            f"Expected a dict with keys 'class' and 'params', but got {pretty.pretty_repr(class_dict)}"
        )

    class_ = class_dict["class"]
    params = class_dict["params"]
    return class_(**params)


def timer_decorator(func):
    import timeit
    import inspect

    def wrapper(*args, **kwargs):
        start = timeit.default_timer() * 1000
        result = func(*args, **kwargs)
        end = timeit.default_timer() * 1000
        eplased_time = round(end - start, 1)
        # logger = logging.getLogger(f"model_learn")
        # logger.info(f"函数 {func.__name__} 执行时间: {eplased_time} ms")
        # print(f"函数 {inspect.getfile(func)}:[{func.__name__}] 执行时间: {eplased_time} ms")
        return result

    return wrapper


def fix_print():
    import builtins, os

    origin_print = builtins.print

    def custom_print(*args, **kwargs):
        import datetime
        import inspect

        timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        prefix = f"[{timestamp}] [{os.path.basename(caller.filename)}:{caller.lineno}]"
        origin_print(prefix, *args, **kwargs)

    builtins.print = custom_print


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


# 获取项目根目录
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Summary:
    step_dict = defaultdict(lambda: 0)
    log_dir = os.path.join(get_project_root(), "logs", "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer: SummaryWriter = SummaryWriter(log_dir)

    @classmethod
    def setpath(cls, subpath):
        log_dir = os.path.join(get_project_root(), "logs", "tensorboard", subpath)
        os.makedirs(log_dir, exist_ok=True)
        cls.writer: SummaryWriter = SummaryWriter(log_dir)

    @classmethod
    def add_scalar(cls, key, value, global_step=None, wall_time=None):
        if not global_step:
            cls.step_dict[key] = cls.step_dict[key] + 1
            global_step = cls.step_dict[key]
        cls.writer.add_scalar(key, value, global_step, wall_time)
