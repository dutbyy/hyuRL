import gymnasium
import numpy as np
from drill.pipeline.interface import ObsData, ActionData, History
from drill import summary
from typing import Dict, List, Any, Tuple
import logging
from absl import flags
from pysc2 import maps, run_configs
from pysc2.lib import point, protocol
from s2clientprotocol import sc2api_pb2 as sc_pb
from .sc2_utils import DIFFICULTIES, RACES, UnitType

from . import sc2_utils
from .sc2_utils import UnitType
from s2clientprotocol import sc2api_pb2 as sc_pb



MAP_MIN_X = 22.
MAP_MAX_X = 44.
MAP_MIN_Y = 20.
MAP_MAX_Y = 36.

N_GRID = 10    # 将地图全局划分为 n_grids * n_grids 的格子
X_GRID_SIZE = (MAP_MAX_X - MAP_MIN_X) / N_GRID
Y_GRID_SIZE = (MAP_MAX_Y - MAP_MIN_Y) / N_GRID

def getLogger(env_id):
    log_name = env_id if isinstance(env_id, str) else f"env-{env_id}"
    logger = logging.getLogger(f"env-{env_id}")
    logger.setLevel(20)
    formatter = logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    try:
        import os

        os.system("mkdir -p /job/logs/user_log/")
        handler = logging.FileHandler(f"/job/logs/user_log/{log_name}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except:
        pass
    return logger


class MinigameEnv:

    def __init__(self,
                 env_id: int,
                 extra_info: Dict,
                 step_mul: int = 2,
                 max_episode_len: int = 400,
                 render: bool = False):
        """初始化 MinigameEnv

        Parameters
        ----------
        step_mul : int, optional
            等价于跳过 step_mul 个帧, by default 2。可参考：
            https://github.com/deepmind/pysc2/blob/master/docs/environment.md
        max_episode_len : int, optional
            回合最大步数, by default 400
        render : bool, optional
            是否渲染, by default False
        """
        self._step_mul = step_mul
        self._max_episode_len = max_episode_len
        self._render = render

        flags.FLAGS(['scc_game_runner'])
        run_config = run_configs.get()
        self.sc2_proc = run_config.start(want_rgb=False)
        controller = self.sc2_proc.controller

        map = maps.get('DefeatZerglingsAndBanelings')
        create = sc_pb.RequestCreateGame(local_map=sc_pb.LocalMap(map_path=map.path,
                                                                  map_data=run_config.map_data(
                                                                      map.path)),
                                         realtime=False,
                                         disable_fog=False)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=RACES['P'], difficulty=DIFFICULTIES["7"])
        controller.create_game(create)

        screen_size_px = point.Point(64, 64)
        minimap_size_px = point.Point(64, 64)
        interface = sc_pb.InterfaceOptions(raw=True,
                                           score=True,
                                           feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        join = sc_pb.RequestJoinGame(race=RACES['P'], options=interface)
        controller.join_game(join)

        self._controller = controller
        if render:
            from pysc2.lib import renderer_human
            self._renderer_human = renderer_human.RendererHuman(render_sync=True)
            game_info = self._controller.game_info()
            static_data = self._controller.data()
            self._renderer_human.init(game_info, static_data)
        self._step_count = 0
        self._last_marine_num = 0

    def reset(self):
        if self._step_count > 0:
            self._controller.restart()
        self._step_count = 0
        self._controller.step(5)
        obs = self._controller.observe()
        all_units = self._get_units(obs.observation)
        self._last_marine_num = len(all_units[UnitType.TERRAN_MARINE])
        self.render(obs)
        return {"minigame_agent": ObsData(obs.observation)}

    def step(self, command_dict: Dict[str, List[sc_pb.Action]]) -> Tuple[Any, bool, Dict]:
        """MinigameEnv 执行一步

        Parameters
        ----------
        command_dict : Dict[str, List[sc_pb.Action]]
            Dict 中的 str 表示环境中每个参与者的 name, 该环境中也是 drill 层 agent 的 name。

        Returns
        -------
        Tuple[Any, bool, Dict]
            obs.observation, done, {}
        """
        self._step_count += 1
        if self._controller.status != protocol.Status.ended:
            command = list(command_dict.values())[0]["cmds"]
            self._controller.actions(sc_pb.RequestAction(actions=command))
        self._controller.step(self._step_mul)
        obs = self._controller.observe()
        self.render(obs)
        done = self._is_done(obs.observation)
        return {"minigame_agent": ObsData(obs.observation)}, done

    def render(self, obs):
        if self._render:
            self._renderer_human.render(obs)

    def _is_done(self, obs: Any) -> bool:
        """episode 是否结束

        Parameters
        ----------
        obs : Any
            原生环境返回的 observation

        Returns
        -------
        bool
            done
        """
        if self._controller.status == protocol.Status.ended:
            return True
        if self._step_count > self._max_episode_len:
            return True

        all_units = self._get_units(obs)
        marine_num = len(all_units[UnitType.TERRAN_MARINE])

        # 我方全灭
        if marine_num == 0:
            return True
        # 敌方全灭
        if len(all_units[UnitType.ZERG_BANELING]) + len(all_units[UnitType.ZERG_ZERGLING]) == 0:
            return True

        # 为了解决4.10环境中的bug，敌方全灭的那一帧捕捉不到
        if marine_num > self._last_marine_num:
            return True
        self._last_marine_num = marine_num

        return False

    def _get_units(self, obs: Any) -> Dict[UnitType, Dict]:
        """获取所有的单元

        Parameters
        ----------
        obs : Any
            环境返回的原生 observation

        Returns
        -------
        Dict[UnitType, Dict]
            all_units
        """
        all_units = {
            UnitType.TERRAN_MARINE: {},
            UnitType.ZERG_BANELING: {},
            UnitType.ZERG_ZERGLING: {}
        }
        for unit in obs.raw_data.units:
            if unit.unit_type == UnitType.TERRAN_MARINE:
                all_units[UnitType.TERRAN_MARINE][unit.tag] = unit
            elif unit.unit_type == UnitType.ZERG_BANELING:
                all_units[UnitType.ZERG_BANELING][unit.tag] = unit
            elif unit.unit_type == UnitType.ZERG_ZERGLING:
                all_units[UnitType.ZERG_ZERGLING][unit.tag] = unit
        return all_units

def _get_units(obs: Any) -> Dict[UnitType, Dict]:  # TODO all_units ? Dict[int, Dict]
    all_units = {UnitType.TERRAN_MARINE: {}, UnitType.ZERG_BANELING: {}, UnitType.ZERG_ZERGLING: {}}
    for unit in obs.raw_data.units:
        if unit.unit_type == UnitType.TERRAN_MARINE:
            all_units[UnitType.TERRAN_MARINE][unit.tag] = unit
        elif unit.unit_type == UnitType.ZERG_BANELING:
            all_units[UnitType.ZERG_BANELING][unit.tag] = unit
        elif unit.unit_type == UnitType.ZERG_ZERGLING:
            all_units[UnitType.ZERG_ZERGLING][unit.tag] = unit
    return all_units


def _make_common_feature(all_units: Dict[str, Dict]) -> Dict[str, int]:
    common_feature_dict = {}
    common_feature_dict["marine_alive_number"] = len(all_units[UnitType.TERRAN_MARINE]) / 10.0
    common_feature_dict["baneling_alive_number"] = len(all_units[UnitType.ZERG_BANELING]) / 10.0
    common_feature_dict["zergling_alive_number"] = len(all_units[UnitType.ZERG_ZERGLING]) / 10.0
    return common_feature_dict


def _make_enemy_features(all_units: Dict[str, Dict]) -> List[Dict]:
    units_features = []
    for unit_type in [UnitType.ZERG_BANELING, UnitType.ZERG_ZERGLING]:
        units = sorted(all_units[unit_type].values(), key=lambda u: u.tag)

        for unit in units:
            unit_feature_dict = {}
            health = unit.health / unit.health_max
            unit_feature_dict["health"] = health

            x_loc = (unit.pos.x - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)
            y_loc = (unit.pos.y - MAP_MIN_Y) / (MAP_MAX_Y - MAP_MIN_Y)
            unit_feature_dict["x_loc"] = x_loc
            unit_feature_dict["y_loc"] = y_loc
            unit_feature_dict["type"] = 0 if unit_type == UnitType.ZERG_BANELING else 1
            units_features.append(unit_feature_dict)
    return units_features


def _make_mask_features(all_units: Dict[str, Dict], units_features: List[Dict]):
    baneling_exists = len(all_units[UnitType.ZERG_BANELING]) != 0
    masks = []
    for unit_feature_dict in units_features:
        if unit_feature_dict["type"] == 0:
            masks.append(1)
        elif unit_feature_dict["type"] == 1 and not baneling_exists:
            masks.append(1)
        else:
            masks.append(0)
    if len(masks) < 10:
        paddings = [0.0 for _ in range(10 - len(masks))]
        masks.extend(paddings)
    return {"attack_target": masks}


def _make_marine_features(all_units: Dict[str, Dict]) -> List[Dict]:
    units = sorted(all_units[UnitType.TERRAN_MARINE].values(), key=lambda u: u.tag)

    units_features = []
    for unit in units:
        unit_feature_dict = {}
        health = unit.health / unit.health_max
        unit_feature_dict["health"] = health

        x_loc = (unit.pos.x - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)
        y_loc = (unit.pos.y - MAP_MIN_Y) / (MAP_MAX_Y - MAP_MIN_Y)
        unit_feature_dict["x_loc"] = x_loc
        unit_feature_dict["y_loc"] = y_loc
        units_features.append(unit_feature_dict)
    return units_features


class PipelineImplement:
    @staticmethod
    def feature_handler(data: ObsData, history: History):
        all_units = _get_units(data.obs)
        marine_features = _make_marine_features(all_units)
        enemy_features = _make_enemy_features(all_units)
        common_feaure_dict = _make_common_feature(all_units)
        enemy_mask_features = _make_mask_features(all_units, enemy_features)
        name2feature = {
            "marine": marine_features,
            "enemy": enemy_features,
            "common": common_feaure_dict,
            "enemy_mask": enemy_mask_features,
        }
        history.agents_history[data.agent_name].push({"obs": data.obs})
        return name2feature

    @staticmethod
    def reward_handler(data: ObsData, history: History):

        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        def score(mean, curr):
            return 4 * sigmoid(0 - np.sqrt(abs(curr - mean)))

        reward = 0
        if data.extra_info_dict["episode_done"]:
            all_units = _get_units(data.obs)
            marine_num = len(all_units[UnitType.TERRAN_MARINE])
            baneling_num = len(all_units[UnitType.ZERG_BANELING])
            zergling_num = len(all_units[UnitType.ZERG_ZERGLING])
            if marine_num > 0:
                marine_health_sum = sum([unit.health for unit in all_units[UnitType.TERRAN_MARINE].values()])
                marine_mean_health = marine_health_sum / marine_num
                health_diff_score = np.mean(
                    [score(marine_mean_health, unit.health) for unit in all_units[UnitType.TERRAN_MARINE].values()]
                )
                if baneling_num + zergling_num == 0:  # Win
                    health_diff_score = 1.0 + health_diff_score / 2  # in range [1, 2]
                    health_sum_score = score(np.sqrt(45 * marine_num), np.sqrt(marine_health_sum))  # in range [0, 2]
                    reward += health_diff_score + health_sum_score
                else:
                    health_diff_score = health_diff_score - 2  # in range [-2, 0]
                    health_sum_score = 1.0 + score(np.sqrt(45 * marine_num), np.sqrt(marine_health_sum)) / 2
                    # in range [-1, 0]
                    reward += (health_diff_score * health_sum_score) / 4
        history.agents_history[data.agent_name].last()["reward"] = reward

        return reward

    @staticmethod
    def action_handler(data: ActionData, history: History):
        last_obs = history.agents_history[data.agent_name].last()["obs"]
        all_units = _get_units(last_obs)
        marines = all_units[UnitType.TERRAN_MARINE]
        marine_tags = sorted(marines.keys())
        selected_marines = data.action["selected_units"][: len(marine_tags)]
        selected_marines = [True if selected else False for selected in selected_marines]
        selected_marine_tags = list(np.array(marine_tags)[selected_marines])

        cmds = []
        # local move

        if data.action["meta"] == 0:
            for marine_tag in selected_marine_tags:
                marine = marines[marine_tag]
                world_x = marine.pos.x + 2 * (data.action["pos_x"] - 1)
                world_y = marine.pos.y + 2 * (data.action["pos_y"] - 1)
                sc2_cmd = sc2_utils.unit_command_pos(
                    sc_pb.Action(), sc2_utils.Ability.MOVE, world_x, world_y, [marine_tag], False
                )
                cmds.append(sc2_cmd)
            data.action.pop("target")
        else:  # 进攻 enemy
            banelines = all_units[UnitType.ZERG_BANELING]
            zerglings = all_units[UnitType.ZERG_ZERGLING]
            if len(banelines) + len(zerglings) > 0:
                baneline_tags = list(sorted(banelines.keys()))
                zergling_tags = list(sorted(zerglings.keys()))
                enemy_tags = baneline_tags + zergling_tags

                target_enemy_tag = enemy_tags[data.action["target"]]
                sc2_cmd = sc2_utils.unit_command_tag(
                    sc_pb.Action(), sc2_utils.Ability.ATTACK, target_enemy_tag, selected_marine_tags, False
                )
                data.action.pop("pos_x")
                data.action.pop("pos_y")
                cmds.append(sc2_cmd)
        data.action["cmds"] = cmds
        return data
