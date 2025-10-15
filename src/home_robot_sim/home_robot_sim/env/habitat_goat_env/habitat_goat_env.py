from typing import Any, Dict, List, Optional, Tuple, Union, cast

import os
import habitat
import json
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import home_robot
from home_robot.perception.constants import (
    SemanticCategoryMapping,
    HabitatObjNav2022Categories,
    GoatCategories,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv
from home_robot_sim.env.habitat_goat_env.visualizer import Visualizer
from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)

from home_robot_sim.env.habitat_goat_env.visualizer import Visualizer
from home_robot.perception.constants import hm3d_raw_to_hm3d, hm3d_raw_to_mp3d

from home_robot.utils.logger import get_logger

logger = get_logger()


class HabitatGoatEnv(HabitatEnv):
    semantic_category_mapping: SemanticCategoryMapping

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.config = config
        self.min_depth = (
            config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.min_depth
        )
        self.max_depth = (
            config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.max_depth
        )
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.task_type = config.habitat.task.type
        self.timestep = 0
        self.current_episode = None

        self.episodes_data_path = config.habitat.dataset.data_path

        if config.AGENT.SEMANTIC_MAP.semantic_categories == "dataset":
            dataset_type = config.habitat.dataset.type
            if dataset_type == "Goat-v1":
                self.semantic_category_mapping = GoatCategories(self.fetch_vocabulary())
            elif dataset_type == "ObjectNav-v1":
                self.semantic_category_mapping = HabitatObjNav2022Categories()
            else:
                raise NotImplementedError
        else:
            #! This is when we want to use custom vocabularies, and not only possible goals
            raise NotImplementedError

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )

        self.visualizer = Visualizer(config, self.semantic_category_mapping)
        self.imagenav_visualizer = NavVisualizer(config, self.semantic_category_mapping)

        if not self.ground_truth_semantics:
            self.init_perception_module()

    def fetch_vocabulary(self):
        if self.config.habitat.dataset.type == "Goat-v1":
            vocabulary = sorted(
                {
                    "_".join(goal.split(" "))
                    for goal in self.habitat_env._dataset.all_categories
                }
            )
            logger.info("Vocabulary: {vocabulary}")
        else:
            raise NotImplementedError

        return vocabulary

        # # TODO: get open set vocabulary
        # vocabulary = []
        # for goal in goals:
        #     vocabulary.append(goal["target"])
        #     if "landmarks" in goal.keys():
        #         vocabulary += goal["landmarks"]
        # return set(vocabulary)

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode
        self.active_task_idx = 0

        if not self.ground_truth_semantics:
            self.init_perception_module()

        self._last_obs = self._preprocess_obs(habitat_obs)
        self.visualizer.reset()
        self.imagenav_visualizer.reset()

        self.scene_id = self.habitat_env.current_episode.scene_id.split("/")[-1].split(
            "."
        )[0]
        self.episode = self.habitat_env.current_episode
        self.episode_id = self.episode.episode_id

        self.current_task_idx = (
            self.habitat_env.task.current_task_idx if self.task_type == "Goat-v1" else 0
        )
        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
    
    def reset_visualization(self):
        self.visualizer.set_vis_dir(
            self.scene_id, f"{self.episode_id}_{self.current_task_idx}"
        )
        self.imagenav_visualizer.set_vis_dir(
            f"{self.scene_id}_{self.episode_id}_{self.current_task_idx}"
        )

    def init_perception_module(self, vocabulary=None):
        from home_robot.perception.detection.detic.detic_perception import (
            DeticPerception,
        )

        # all_ovon_categories_path = "/srv/flash1/rramrakhya3/fall_2023/goat/data/hm3d_meta/ovon_categories_final_split.json"
        # with open(all_ovon_categories_path, "r") as f:
        #     all_ovon_categories = json.load(f)

        # # all_ovon_categories = [y for x in all_ovon_categories.values() for y in x if type(y) == str]
        # all_ovon_categories = sorted(list(set(all_ovon_categories["val_seen"])))

        # all_ovon_categories = ["_".join(x.split(" ")) for x in all_ovon_categories]

        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary="," + ",".join(self.semantic_category_mapping.vocabulary),
            sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        )

        # self.segmentation = MaskRCNNPerception(
        #     sem_pred_prob_thr=0.9,
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        # )

        # from home_robot.perception.detection.grounded_sam.ram_perception import RAMPerception

        # self.segmentation = RAMPerception(
        #     custom_vocabulary=".",
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        #     verbose=False,
        #     # **module_kwargs
        # )

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])
        if habitat_obs.get("multigoal") is None:
            goals = self._preprocess_goals(
                [{"category": self.current_episode.object_category}] # we can also read from obs[objectgoal]
            )
        else:
            goals = self._preprocess_goals(habitat_obs["multigoal"])
        obs = home_robot.core.interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={
                "tasks": goals,
                "top_down_map": self.get_episode_metrics().get("goat_top_down_map"),
            },
            camera_pose=None,
            third_person_image=None,
        )
        obs = self._preprocess_semantic(obs, habitat_obs["semantic"])
        return obs

    #! THis is inside habitat goat env:
    def visualize_semantic_with_labels(
        self,
        semantic_array: np.ndarray,
        palette: list,
        label_min_pixels: int = 50,
        font_scale: float = 0.4,
        thickness: int = 1,
        postfix: str = "",
    ):
        """
        Visualizes a semantic map with color palette and overlays ID numbers on each region.

        Args:
            semantic_array (np.ndarray): 2D array of shape (H, W) with semantic IDs.
            palette (list): A flat list of RGB values, e.g., [R0, G0, B0, R1, G1, B1, ...].
            save_path (str): File path to save the resulting image (e.g., 'out.png').
            label_min_pixels (int): Minimum number of pixels required to label a region.
            font_scale (float): Font scale for the overlaid text.
            thickness (int): Text thickness.
        """
        import os
        import numpy as np
        import cv2
        from PIL import Image

        palette_img = Image.new("P", (semantic_array.shape[1], semantic_array.shape[0]))
        palette_img.putpalette(palette)
        palette_img.putdata(semantic_array.flatten().astype(np.uint8))
        palette_img = palette_img.convert("RGB")

        semantic_map_cv = np.array(palette_img)[
            :, :, ::-1
        ].copy()  # RGB -> BGR, and make it OpenCV-safe

        unique_ids = np.unique(semantic_array)
        for sid in unique_ids:
            coords = np.argwhere(semantic_array == sid)
            center_coord = coords[len(coords) // 2]
            if len(coords) < label_min_pixels:
                continue
            y, x = center_coord[:2]
            cv2.putText(
                semantic_map_cv,
                str(int(sid)),
                (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        if not self.visualizer.vis_dir is None:
            save_path = os.path.join(
                self.visualizer.vis_dir, f"{self.timestep}_0.sem_input{postfix}.png"
            )
            cv2.imwrite(save_path, semantic_map_cv)

    def _preprocess_semantic(
        self, obs: home_robot.core.interfaces.Observations, habitat_semantic: np.ndarray
    ) -> home_robot.core.interfaces.Observations:
        habitat_semantic = habitat_semantic[:,:,-1]
        if self.ground_truth_semantics:
            # * shape of habitat_semantic: (H, W, 1), shape of obs.semantic: (H, W) (only numbers change)

            # self.visualize_semantic_with_labels(
            #     semantic_array=habitat_semantic,
            #     palette=self.semantic_category_mapping.map_color_palette,
            #     postfix="_gt",
            # )
            # NEW
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id
            )
            max_id = habitat_semantic.max()
            if max_id >= len(instance_id_to_category_id):
                #! This can happen, because of the problems in labels. For some reason, the label is not in the list of all objects. This is not a problem of the code.
                logger.warning(f"Warning: semantic ID {max_id} exceeds mapping size {len(instance_id_to_category_id)}")
            habitat_semantic[habitat_semantic >= len(instance_id_to_category_id)] = 0
            obs.semantic = instance_id_to_category_id[habitat_semantic]
            obs.task_observations["instance_map"] = habitat_semantic + 1
            # self.visualize_semantic_with_labels(
            #     semantic_array=obs.semantic + 10,
            #     palette=self.semantic_category_mapping.map_color_palette,
            # )

            # OLD
            # obs.semantic = np.vectorize(lambda x: self.hm3d_mapping.get(x, 0))(habitat_semantic)[..., 0]
            # obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1
            # self.visualize_semantic_with_labels(
            #     semantic_array=obs.semantic+10,
            #     palette=self.semantic_category_mapping.map_color_palette,
            # )
        else:
            obs = self.segmentation.predict(obs)

        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goals(self, goals):
        for goal_v in goals:
            goal_v["semantic_id"] = self.semantic_category_mapping.goal_name_to_goal_id[
                "_".join(goal_v["category"].split(" "))
            ]
            if goal_v.get("image") is not None:
                goal_v["type"] = "imagenav"
            elif goal_v.get("description") is not None:
                goal_v["type"] = "languagenav"
            else:
                goal_v["type"] = "objectnav"

        return goals

    def _preprocess_action(self, action: home_robot.core.interfaces.Action) -> int:

        if type(action) == int:
            return action

        discrete_action = cast(
            home_robot.core.interfaces.DiscreteNavigationAction, action
        )
        return HabitatSimActions[discrete_action.name.lower()]

    def _process_info(self, info: Dict[str, Any], agent_id=None) -> Any:
        obs = self.get_observation()
        if isinstance(obs, list):
            obs = obs[agent_id]
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        if (
            self.task_type == "Goat-v1" and
            current_task["type"] == "image"
        ):
            info["rgb_frame"] = obs.rgb
            info["semantic_frame"] = obs.semantic
            info["last_goal_image"] = current_task["image"]
            info["last_collisions"] = {"is_collision": False}
            info["last_td_map"] = obs.task_observations.get("top_down_map")
            self.imagenav_visualizer.visualize(**info)
        else:
            goal_text_desc = {
                x: y
                for x, y in current_task.items()
                if x != "image"
            }
            info["goal_name"] = str(goal_text_desc)
            info["semantic_frame"] = obs.task_observations["semantic_frame"]
            info["third_person_image"] = obs.third_person_image
            info["top_down_map"] = obs.task_observations.get("top_down_map")
            info["agent_id"] = agent_id
            self.visualizer.visualize(**info)


    def apply_action(
        self,
        action: List[home_robot.core.interfaces.Action],
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[home_robot.core.interfaces.Observations] = None,
    ):
        super().apply_action(action, info, prev_obs)
        self.current_task_idx = (
            self.habitat_env.task.current_task_idx if self.task_type == "Goat-v1" else 0
        )


class MultiAgentHabitatGoatEnv(HabitatGoatEnv):
    semantic_category_mapping: SemanticCategoryMapping

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env, config)

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        if habitat_obs.get("multigoal") is None:
            # I can also get habitat_obs["objectgoal"] here, but it is not compatible, because multigoal returns text category
            goals = self._preprocess_goals(
                [{"category": self.current_episode.object_category}]
            )
        else:
            goals = self._preprocess_goals(habitat_obs["multigoal"])

        observations = []
        for agent_id in range(self.config.NUM_AGENTS):
            agent_obs = habitat_obs[agent_id]
            depth = self._preprocess_depth(agent_obs[f"depth"])
            obs = home_robot.core.interfaces.Observations(
                rgb=agent_obs[f"rgb"],
                depth=depth,
                compass=habitat_obs["compass"][agent_id],
                gps=self._preprocess_xy(habitat_obs["gps"][agent_id]),
                task_observations={
                    "tasks": goals,
                    "top_down_map": self.get_episode_metrics().get("goat_top_down_map"),
                },
                camera_pose=None,
                third_person_image=None,
            )
            obs = self._preprocess_semantic(obs, agent_obs[f"semantic"])
            observations.append(obs)
        return observations

    def _preprocess_action(self, actions: home_robot.core.interfaces.Action) -> int:

        if type(actions[0]) == int:
            return actions

        discrete_actions = []
        for action in actions:
            discrete_actions.append(
                cast(home_robot.core.interfaces.DiscreteNavigationAction, action)
            )
        return {
            i: HabitatSimActions[discrete_action.name.lower()]
            for i, discrete_action in enumerate(discrete_actions)
        }

    def _process_info(self, infos: List[Dict[str, Any]]) -> Any:
        for i, info in enumerate(infos):
            super()._process_info(info, i)