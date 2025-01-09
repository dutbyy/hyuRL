from typing import Any, List

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

RACES = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

DIFFICULTIES = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}


class UnitType:
    TERRAN_MARINE = 48
    ZERG_BANELING = 9
    ZERG_ZERGLING = 105


class Ability(object):
    MOVE = 16
    ATTACK = 3674


def unit_command_pos(action: sc_pb.Action, ability_id: Ability.MOVE, x: Any, y: Any,
                     unit_tags: np.ndarray, queue: bool) -> sc_pb.Action:
    """移动

    Parameters
    ----------
    action : sc_pb.Action
        原生环境中的 action
    ability_id : Ability.MOVE
        MOVE
    x : Any
        横坐标
    y : Any
        纵坐标
    unit_tags : np.ndarray
        被移动的 unit
    queue : bool
        是否加入 queue

    Returns
    -------
    sc_pb.Action
        原生环境可以识别的 action
    """

    cmd = action.action_raw.unit_command
    cmd.ability_id = ability_id
    cmd.target_world_space_pos.x = x
    cmd.target_world_space_pos.y = y
    cmd.unit_tags[:] = unit_tags
    cmd.queue_command = queue
    return action


def unit_command_tag(action: sc_pb.Action, ability_id: Ability.ATTACK, target_unit_tag: Any,
                     unit_tags: List[np.ndarray], queue: bool) -> sc_pb.Action:
    """攻击 enemy

    Parameters
    ----------
    action : sc_pb.Action
        原生环境中的 action
    ability_id : Ability.ATTACK
        ATTACK
    target_unit_tag : Any
        被进攻的 units
    unit_tags : List[np.ndarray]
        主动进攻的 units
    queue : bool
        是否加入 queue

    Returns
    -------
    sc_pb.Action
        原生环境可以识别的 action
    """
    cmd = action.action_raw.unit_command
    cmd.ability_id = ability_id
    cmd.target_unit_tag = target_unit_tag
    cmd.unit_tags[:] = unit_tags
    cmd.queue_command = queue
    return action
