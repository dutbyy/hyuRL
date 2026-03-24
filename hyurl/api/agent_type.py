from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
class FixedSizeBuffer:

    def __init__(self, size):
        self._deque = deque(maxlen=size)

    def __getitem__(self, index):
        return self._deque[index]

    @property
    def data(self):
        return self._deque

    def push(self, value):
        self._deque.append(value)

    def last(self):
        return self._deque[-1]

    def clear(self):
        self._deque.clear()


@dataclass
class ObsData:
    obs: Any
    extra_info_dict: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""


@dataclass
class ActionData:
    action: Dict[str, np.ndarray]    # action_head name to action
    predict_output: Dict[str, np.ndarray]
    action_mask: Dict[str, np.ndarray] = field(default_factory=dict)    # action_head name to mask
    agent_name: str = ""


class History:

    def __init__(self, maxlen):
        self._global = FixedSizeBuffer(maxlen)
        self._agents = defaultdict(lambda: FixedSizeBuffer(maxlen))

    @property
    def global_history(self):
        return self._global

    @property
    def agents_history(self) -> Dict[str, FixedSizeBuffer]:
        return self._agents

    def clear(self):
        self._global.clear()
        for agent_history in self._agents.values():
            agent_history.clear()



