# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
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
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)


class Task:
    type: str
    goal_semantic_id: int
    goal_image: np.ndarray = None
    goal_image_processed: np.ndarray = None
    goal_image_keypoints: np.ndarray = None
    goal_description: str = None


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map
    Works for tasks: Objectnav, Goat-v1, MultiagentGoat-V1

    """

    def __init__(
        self, config, semantic_category_mapping, agent_id=None, device_id: int = 0
    ):
        if agent_id is None:
            self.is_multiagent = False
        else:
            self.is_multiagent = True

        self.agent_id = agent_id
        self.log = get_logger(agent_id=agent_id)
        self.max_steps = [config.AGENT.max_steps] * 10
        self.task_type = config.habitat.task.type
        self.seq_goals = bool(config.SEQ)

        # self.record_instance_ids = True if "Goat-v1" in self.task_type else False
        self.record_instance_ids = (
            True  # Code doesn't work with False, I can fix this later
        )

        self.instance_memory = None
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config.AGENT.SEMANTIC_MAP.du_scale,
                # debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
            logger=self.log,
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
            record_instance_ids=self.record_instance_ids,
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

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=self.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            # close_frontier_radius=10.0,  #! myTODO: Hardcoded 5
            agent_id=agent_id,
        )
        self.max_subtasks_per_episode = config.ENVIRONMENT.max_subtasks_per_episode

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

        self.subtask_timesteps = None
        self.total_timesteps = None
        self.last_pose = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.communication_radius = config.AGENT.COMMUNICATION.radius
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=","
                + ",".join(self.semantic_category_mapping.vocabulary),
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )
        self.match_memory = True

    def get_subtask_timestep(self) -> int:
        """
        Only has meaning when we have sequential goals and a single agent. Otherwise, it is the total timestep.
        """
        if self.seq_goals:
            return self.subtask_timesteps[self.current_task_idx]
        else:
            return self.total_timesteps

    def reset(self, scene_id, episode_id):
        """Initialize agent state. Reset is at the beginning of a new episode (not each task)."""
        self.total_timesteps = 0
        if self.seq_goals:
            self.subtask_timesteps = [0] * self.max_subtasks_per_episode
        self.last_pose = np.zeros(3)
        self.semantic_map.init_map_and_pose()
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.navigate_to_best = False

        self.current_task_idx = -1
        self.reset_for_next_task()
        self.planner.reset()
        self.matching.step = 0
        self.inst_goal_id = None

        self.stuck_counter = 0
        self.reset_vis_dir(scene_id, episode_id, 0)
        self.last_communication_time = {}

    def handle_stop(self, action):
        self.reset_for_next_task()

    def reset_for_next_task(self) -> None:
        """Reset for a new task, and not a new episode."""
        self.inst_goal_id = None
        if self.seq_goals:
            self.current_task_idx += 1
        self.navigate_to_best = False
        self.match_memory = True
        self.planner.reset_for_next_task()

    def update_steps(self):
        self.total_timesteps += 1
        if self.seq_goals:
            self.subtask_timesteps[self.current_task_idx] += 1
        self.matching.step = self.total_timesteps
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        self.planner.total_timesteps = self.total_timesteps
        self.planner.timestep = self.get_subtask_timestep()

    def update_state(self, obs):
        self._last_obs = obs
        obs_preprocessed, pose_delta = self._preprocess_obs(obs)
        self.update_steps()
        self.log.info(
            f"---------------- Updating state - step:{self.get_subtask_timestep()} ----------------"
        )
        self.log.debug(
            f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB"
        )

        self._update_maps(obs_preprocessed, pose_delta)

        self._search_for_goal()
        if torch.norm(pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = []
        for task_obs in tasks_obs:
            task = Task()
            task.type = task_obs["type"]
            task.goal_semantic_id = task_obs["semantic_id"]
            if task.type == "imagenav":
                task.goal_image = task_obs["image"]
                task.goal_image_processed, task.goal_image_keypoints = (
                    self.matching.get_goal_image_keypoints(task.goal_image)
                )
            elif task.type == "languagenav":
                task.goal_description = task_obs["description"]
            tasks.append(task)
        return tasks

    def act(self, **kwargs) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """Act end-to-end."""
        stuck = False
        if (
            self.get_subtask_timestep() >= self.max_steps[self.current_task_idx]
        ) or self.stuck_counter > 30:
            self.log.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            stuck = True

        action, vis_inputs = self._get_best_action(**kwargs)
        action = self._process_action(action)
        if action["action"] == DiscreteNavigationAction.STOP:
            self.handle_stop(action)

        info = self._get_vis_info(vis_inputs, action)

        return action, info, stuck

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

    def _get_vis_info(self, vis_inputs, action):
        if not self.visualize:
            return None

        is_local = vis_inputs.get("is_local", True)
        obs = self._last_obs
        info = {
            "agent_id": self.agent_id,
            "rgb_frame": obs.rgb[:, :, ::-1],
            "semantic_frame": obs.semantic,
            "top_down_map": obs.task_observations.get("top_down_map"),
            "is_collision": False,  #!myTODO
            "inst_goal_id": self.inst_goal_id,
            "timestep": self.get_subtask_timestep(),
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
        self._get_task_info(obs, action, info)
        return info

    def _get_task_info(self, obs, action, info):
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        info["task_type"] = current_task["type"]
        goal_text_desc = {x: y for x, y in current_task.items() if x != "image"}
        info["caption"] = str(goal_text_desc)
        if current_task["type"] == "imagenav":
            info["goal_image"] = current_task["image"]
        else:
            info["third_person_image"] = obs.third_person_image

        info["caption"] += f" | Action: {action['action']}"

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        if not self.ground_truth_semantics:
            obs = self.segmentation.predict(obs)
            obs.task_observations["instance_frame"] = (
                obs.task_observations["instance_map"] + 1
            )

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device)
        ][
            :, :, 1:
        ]  # one-hot encode and remove background class

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            # * Why using instance_frame which are the raw semantics? To differentiate between objects with diff raw semantics but same category in our ovon classes.
            instances = obs.task_observations["instance_frame"]
            # import os
            # import cv2
            # from home_robot.utils.visualization import visualize_semantic_with_labels
            # visualize_semantic_with_labels(
            #     semantic_array=instances,
            #     palette=self.semantic_category_mapping.map_color_palette,
            #     save_path=os.path.join(self.planner.vis_dir, f"{self.get_subtask_timestep()}_TEMP.instance_ids.png"),
            # )
            instance_ids = np.unique(instances)
            instance_ids, instances_idx = np.unique(instances, return_inverse=True)
            instances_idx = instances_idx.reshape(instances.shape)
            instances = torch.from_numpy(instances_idx).to(self.device)

            # One-hot encode
            instance_frame_onehot = torch.eye(len(instance_ids), device=self.device)[
                instances
            ]
            # visualize_semantic_with_labels(
            #     semantic_array=instances.cpu().numpy(),
            #     palette=self.semantic_category_mapping.map_color_palette,
            #     save_path=os.path.join(self.planner.vis_dir, f"{self.get_subtask_timestep()}_TEMP.temp_ids.png"),
            # )

            obs_preprocessed = torch.cat(
                [obs_preprocessed, instance_frame_onehot], dim=-1
            )
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=rgb.device
        )
        self.last_pose = curr_pose

        assert obs.camera_pose is None

        self.tasks = self._preprocess_tasks(obs.task_observations["tasks"])

        # * preprocessed obs shape is (1, 3+1+num_sem_classes+num_instances, H, W)
        return obs_preprocessed, pose_delta

    def _get_best_action(self, **kwargs):
        task = self.tasks[self.current_task_idx]
        action, vis_input = self.planner.plan(
            self.inst_goal_id,
            task.goal_semantic_id,
        )

        if not action is None:
            return action, vis_input

        self.log.info("No reachable goal/frontier.")

        if self.navigate_to_best:
            self.log.info("Already tried the best match. Stopping")
            return DiscreteNavigationAction.STOP, {}
        self.navigate_to_best = True
        self.log.info("Forcing a match against memory")

        prev_inst_goal_id = self.inst_goal_id
        self.inst_goal_id = self.matching.search_for_goal(
            task,
            True,
            self.semantic_map.global_pose,
            score_thresh=0,
        )
        if self.inst_goal_id is None or self.inst_goal_id == prev_inst_goal_id:
            self.log.info("Best match is the same as the previous one. Stopping")
            return DiscreteNavigationAction.STOP, {}

        action, vis_input = self.planner.plan(
            self.inst_goal_id,
            task.goal_semantic_id,
            fallback_to_frontier=False,
            postfix="_last_shot",
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
        Set value for self.inst_goal_id.
        """
        #! myTODO: Put %10 here, so that we again check with obs every 10 steps, so we might get better matches. Can be more intelligent.
        if not self.inst_goal_id is None and self.get_subtask_timestep() % 10 != 0:
            self.log.info(f"Already found instance goal, not searching anymore.")
        else:
            # Match a goal against every instance in memory the moment the subtask starts.
            # We also search in memory when env is fully explored, but that is handled somewhere else.
            inst_goal_id = self.matching.search_for_goal(
                self.tasks[self.current_task_idx],
                self.match_memory,
                self.semantic_map.global_pose,
            )
            if not inst_goal_id is None:
                # Else, we should not replace, maybe we have previously seen a goal and moving toward it.
                self.inst_goal_id = inst_goal_id

        self.match_memory = False

    def reset_vis_dir(self, scene_id, episode_id, current_task_idx=None):
        if self.seq_goals:
            dir_name = f"{scene_id}_{episode_id}_{current_task_idx}"
        else:
            dir_name = f"{scene_id}_{episode_id}"

        if not self.agent_id is None:
            dir_name = os.path.join(dir_name, f"agent_{self.agent_id}")

        self.planner.set_vis_dir(dir_name)
        self.matching.set_vis_dir(dir_name)
        self.semantic_map.vis_dir = self.planner.vis_dir
        self.semantic_map_module.vis_dir = self.planner.vis_dir

    def _process_action(self, action):
        return {
            "action": action,
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": self.current_task_idx,
            },
        }
