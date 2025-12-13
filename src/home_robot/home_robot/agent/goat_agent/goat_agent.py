# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import psutil
import numpy as np
from pathlib import Path
import home_robot.utils.pose as pu
from .goat_matching import GoatMatching
from typing import Any, Dict, List, Tuple
from home_robot.utils.logger import get_logger
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.maskrcnn.coco_categories import coco_categories

from .goat_agent_module import GoatAgentModule
from .goat_matching import GoatMatching

from home_robot.utils.logger import get_logger
logger = get_logger()

from home_robot.utils.visualization import visualize_map

# For visualizing exploration issues
debug_frontier_map = False
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(
        self, config, semantic_category_mapping, agent_id=None, device_id: int = 0
    ):
        # self.max_steps = config.AGENT.max_steps
        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        self.max_steps = [500] * 10

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config.AGENT.SEMANTIC_MAP.du_scale,
                # debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )
        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")

        self.semantic_category_mapping = semantic_category_mapping
        self.num_sem_categories = semantic_category_mapping.num_sem_categories
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        camera_sensor = config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor
        self.semantic_map_module = Categorical2DSemanticMapModule(
            device=self.device,
            frame_height=camera_sensor.height,
            frame_width=camera_sensor.width,
            camera_height=camera_sensor.position[1],
            hfov=camera_sensor.hfov,
            num_sem_categories=self.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=camera_sensor.max_depth,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=self.instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            exploration_type=config.AGENT.exploration_type,
            gaze_width=(
                40 if config.AGENT.exploration_type == "raycast" else 30
            ),  #! myTODO: Hardcoded 3
            gaze_distance=(
                camera_sensor.max_depth
                if config.AGENT.exploration_type == "raycast"
                else 3
            ),  #! myTODO: Hardcoded 3
            agent_cell_radius=agent_cell_radius,
        )
        self.inst_goal_id = None
        self.inst_goal_found = False


        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=self.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=self.instance_memory,
            # close_frontier_radius=10.0,  #! myTODO: Hardcoded 5
            agent_id=agent_id,
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / config.habitat.simulator.turn_angle)
        else:
            panorama_start_steps = 0

        self.planner = DiscretePlanner(
            turn_angle=config.habitat.simulator.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
            panorama_start_steps=panorama_start_steps,
            instance_memory=self.instance_memory,
            goal_filtering=config.AGENT.SEMANTIC_MAP.goal_filtering,
            semantic_map=self.semantic_map,
            frontier_metric=config.AGENT.frontier_metric,
            agent_id=self.agent_id,
        )

        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.last_pose = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.image_matching_function = self.matching.match_image_to_image
        self.matching_fn = {
            "imagenav": self.image_matching_function,
            "languagenav": self.matching.match_language_to_image,
            "objectnav": None,
        }
        self.communication_radius = config.AGENT.COMMUNICATION.radius
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary="," + ",".join(self.semantic_category_mapping.vocabulary),
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )

    def get_subtask_timestep(self) -> int:
        return self.sub_task_timesteps[self.current_task_idx]

    def reset(self, scene_id, episode_id, current_task_idx):
        """Initialize agent state. Reset is at the beginning of a new episode (not each task)."""
        self.total_timesteps = 0
        self.sub_task_timesteps = [0] * self.max_num_sub_task_episodes
        self.last_pose = np.zeros(3)
        self.semantic_map.init_map_and_pose()
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.navigate_to_best = False

        self.current_task_idx = -1
        self.reset_sub_episode()
        self.planner.reset()
        self.inst_goal_found = False
        self.inst_goal_id = None

        self.stuck_counter = 0
        self._reset_vis_dir(scene_id, episode_id, current_task_idx)

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.inst_goal_found = False
        self.inst_goal_id = None

        self.current_task_idx += 1
        self.navigate_to_best = False
    
    def update_state(self, obs):
        self.current_task = obs.task_observations["tasks"][self.current_task_idx]
        self.total_timesteps = self.total_timesteps + 1
        self.sub_task_timesteps[self.current_task_idx] += 1
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        self.log.info(
            f"---------------- Subtask step {self.get_subtask_timestep()} ----------------"
        )
        self.log.debug(
            f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB"
        )

        obs_preprocessed, pose_delta = self._preprocess_obs(obs)

        self._update_maps(obs_preprocessed, pose_delta)

        self._search_for_goal()
        if torch.norm(pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def communicate(self, neighbors):
        if len(neighbors) == 0:
            return
        self.semantic_map.global_map = (
            self.semantic_map_module.merge_neighbor_maps(
                neighbors, self.semantic_map.global_map
            )
        )
        lmb = self.semantic_map.lmb
        self.semantic_map.local_map[:] = self.semantic_map.global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]]

    def act(self, other_agents=None) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        neighbors = self._get_neighbors(other_agents)
        self.communicate(neighbors)
        stuck = False
        if (
            self.get_subtask_timestep() >= self.max_steps[self.current_task_idx]
        ) or self.stuck_counter > 30:
            self.log.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            stuck = True

        action, vis_inputs = self._get_best_action(neighbors)

        info = self._get_vis_info(vis_inputs)

        return action, info, stuck

    def _get_neighbors(self, other_agents):
        neighbors = []
        if other_agents is None:
            return neighbors
        for agent in other_agents:
            dist = (
                agent.semantic_map.global_pose[:2] - self.semantic_map.global_pose[:2]
            ).norm()
            if dist < self.communication_radius:
                neighbors.append(agent)
                self.log.debug(
                    f"Communicating with agent: {agent.agent_id} with distance {dist}"
                )
        return neighbors

    def _update_maps(self, obs: torch.Tensor, pose_delta: torch.Tensor):
        # * before module call obs.shape is [380+3+1+num_instances]
        # Update map with observations and generate map features
        (
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        ) = self.semantic_map_module(
            obs,
            pose_delta,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )

    def _get_vis_info(self, vis_inputs):
        if not self.visualize:
            return None

        is_local = vis_inputs.get("is_local", True)
        info = {
            "inst_goal_id": self.inst_goal_id,
            "timestep": self.get_subtask_timestep(),
            "total_timesteps": self.total_timesteps,
            "explored_map": self.semantic_map.get_explored_map(is_local),
            "obstacle_map": self.semantic_map.get_obstacle_map(is_local),
            "semantic_map_1D": self.semantic_map.get_semantic_map_1D(is_local),
            "frontier_map": self.semantic_map.get_frontier_map(is_local),
            "been_close_map": self.semantic_map.get_been_close_map(is_local),
            "visited_map": self.semantic_map.get_visited_map(is_local),
            "global_pose": self.semantic_map.global_pose,
            "lmb": self.semantic_map.lmb,
            "instance_memory": self.instance_memory,
            **vis_inputs,
        }

        return info

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        if not self.ground_truth_semantics:
            obs = self.segmentation.predict(obs)
            obs.task_observations["instance_frame"] = obs.task_observations["instance_map"] + 1


        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        # * Semantics becomes (W,H,NumClasses) which NumClasses is read from the config files, and is 380. Note that because I am using less classes (52 in all_ovon_categires) most of these layers are zero and actually useless.
        # * Maybe I should change the config. But nevertheles, this works even with 380.
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device)
        ][
            :, :, 1:
        ]  # one-hot encode and remove background class

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            # * Why using instance_frame which are the raw semantics? To differentiate between objects with diff raw semantics but same category in our ovon classes.
            instances = obs.task_observations["instance_frame"]
            # first create a mapping to 1, 2, 3, ..., num_instances
            instance_ids = np.unique(instances)
            instance_ids, instances_idx = np.unique(instances, return_inverse=True)
            instances_idx = instances_idx.reshape(instances.shape)
            instances = torch.from_numpy(instances_idx).to(self.device)

            # One-hot encode
            instance_frame_onehot = torch.eye(len(instance_ids), device=self.device)[
                instances
            ]

            obs_preprocessed = torch.cat(
                [obs_preprocessed, instance_frame_onehot], dim=-1
            )
            # import os
            # import cv2
            # from home_robot.utils.visualization import visualize_semantic_with_labels
            # visualize_semantic_with_labels(
            #     semantic_array=instances.cpu().numpy(),
            #     palette=self.semantic_category_mapping.map_color_palette,
            #     save_path=os.path.join(self.planner.vis_dir, f"{self.get_subtask_timestep()}_TEMP.instance_input.png"),
            # )
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=rgb.device
        )
        self.last_pose = curr_pose

        assert obs.camera_pose is None

        # * preprocessed obs shape is (1, 3+1+num_sem_classes+num_instances, H, W)
        return obs_preprocessed, pose_delta

    def _match_against_current_frame(self):
        image_goal = None
        language_goal = None

        if self.current_task["type"] == "imagenav":
            if self.goal_image is None:
                img_goal = self.current_task["image"]
                self.goal_image, self.goal_image_keypoints = (
                    self.matching.get_goal_image_keypoints(img_goal)
                )
                # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)
            image_goal = self.goal_image

        elif self.current_task["type"] == "languagenav":
            language_goal = self.current_task["description"]

        confidences, frame_matches_instance_ids = (
            self.matching.get_matches_against_current_frame(
                self.matching_fn[self.current_task["type"]],
                self.total_timesteps,
                image_goal=image_goal,
                goal_image_keypoints=self.goal_image_keypoints,
                language_goal=language_goal,
                categories=[self.current_task["semantic_id"]],
                use_full_image=False,
                global_pose=self.semantic_map.global_pose,
            )
        )

        return confidences, frame_matches_instance_ids

    def _match_against_memory(self):
        task_type = self.current_task["type"]
        self.log.info("--------Matching against memory!--------")
        image_goal = None
        language_goal = None
        goal_image_keypoints = None
        if task_type == "languagenav":
            language_goal = self.current_task["description"]
        elif task_type == "imagenav":
            image_goal = self.goal_image
            goal_image_keypoints = self.goal_image_keypoints
        (
            mem_match_confidences,
            mem_match_instance_ids,
        ) = self.matching.get_matches_against_memory(
            self.matching_fn[task_type],
            self.total_timesteps,
            language_goal=language_goal,
            image_goal=image_goal,
            goal_image_keypoints=goal_image_keypoints,
            use_full_image=True,
            categories=[self.current_task["semantic_id"]],
            global_pose=self.semantic_map.global_pose,
        )

        if len(mem_match_confidences) > 0:
            stats = {
                i: {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "all": scores.flatten().tolist(),
                }
                for i, scores in zip(mem_match_instance_ids, mem_match_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_{task_type}_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)
        return mem_match_confidences, mem_match_instance_ids

    def _get_best_action(self, neighbors):
        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            self.current_task["semantic_id"],
            neighbors=neighbors,
        )

        if not action is None:
            return action, vis_input

        self.log.info("No reachable goal/frontier.")

        if self.navigate_to_best:
            self.log.info("Already tried the best match. Stopping")
            return DiscreteNavigationAction.STOP, {}
        self.navigate_to_best = True
        self.log.info("Forcing a match against memory")

        mem_match_confidences, mem_match_instance_ids = self._match_against_memory()
        if not len(mem_match_confidences) > 0:
            self.log.info("No match found in memory. Stopping")
            return DiscreteNavigationAction.STOP, {}
        prev_inst_goal_id = self.inst_goal_id
        (
            self.inst_goal_found,
            self.inst_goal_id,
        ) = self.matching.get_best_inst_goal(
            mem_match_confidences=mem_match_confidences,
            mem_match_instance_ids=mem_match_instance_ids,
            score_thresh=0,
        )
        assert self.inst_goal_found == True
        if self.inst_goal_id == prev_inst_goal_id:
            self.log.info("Best match is the same as the previous one. Stopping")
            return DiscreteNavigationAction.STOP, {}

        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            self.current_task["semantic_id"],
            fallback_to_frontier=False,
            postfix="_last_shot",
            neighbors=neighbors,
        )

        if action is None:
            self.log.info("Fully explored and no path to our best match. Stopping")
            return DiscreteNavigationAction.STOP, {}

        self.log.info("Found a path to the last shot goal. Navigating to it.")
        return action, vis_input

    @torch.no_grad()
    def _search_for_goal(self):
        """
        Searches for goal in current observation, and also in memory if it is the first timestep of the task.
        Set values for self.inst_goal_found and self.inst_goal_id.
        """
        #! myTODO: Put %10 here, so that we again check with obs every 10 steps, so we might get better matches. Can be more intelligent
        if self.inst_goal_found and self.get_subtask_timestep() % 10 != 0:
            self.log.info(f"Already found instance goal, not searching anymore.")
        else:
            # Match a goal against every instance in memory the moment the subtask starts.
            # We also search in memory when env is fully explored, but that is handled somewhere else.
            mem_match_confidences, mem_match_instance_ids = (
                self._match_against_memory()
                if self.get_subtask_timestep() == 0
                else ([], [])
            )
            obs_match_confidences, obs_match_instance_ids = (
                self._match_against_current_frame()
            )
            self.log.debug(
                f"candidate matches in memory: {len(mem_match_confidences)}, candidate matches in observation: {len(obs_match_confidences)}"
            )
            if len(mem_match_confidences) > 0 or len(obs_match_confidences) > 0:
                (
                    self.inst_goal_found,
                    self.inst_goal_id,
                ) = self.matching.get_best_inst_goal(
                    obs_match_confidences,
                    obs_match_instance_ids,
                    mem_match_confidences=mem_match_confidences,
                    mem_match_instance_ids=mem_match_instance_ids,
                    score_thresh=self._score_thresh(),
                )

    def _reset_vis_dir(self, scene_id, episode_id, current_task_idx):
        self.planner.set_vis_dir(scene_id, f"{episode_id}_{current_task_idx}")
        self.matching.set_vis_dir(f"{scene_id}_{episode_id}_{current_task_idx}")
        self.semantic_map.vis_dir = self.planner.vis_dir
        self.semantic_map_module.vis_dir = self.planner.vis_dir

    def _score_thresh(self):
        task_type = self.current_task["type"]
        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0
