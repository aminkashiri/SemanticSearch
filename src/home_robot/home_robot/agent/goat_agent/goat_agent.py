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
from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)

logger = get_logger()


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        # self.max_steps = config.AGENT.max_steps
        # self.max_steps = [500, 500, 500, 500, 500]
        self.max_steps = [500] * 10
        # self.max_steps = [500, 400, 300, 200, 200, 200, 200, 200, 200, 200, 200]
        # self.max_steps = [400, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )

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
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=config.AGENT.SEMANTIC_MAP.max_depth,
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
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=self.instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            exploration_type=config.AGENT.SEMANTIC_MAP.exploration_type,
            gaze_width=(
                40 if config.AGENT.SEMANTIC_MAP.exploration_type == "raycast" else 30
            ),  #! myTODO: Hardcoded 3
            gaze_distance=(
                config.AGENT.SEMANTIC_MAP.max_depth
                if config.AGENT.SEMANTIC_MAP.exploration_type == "raycast"
                else 3
            ),  #! myTODO: Hardcoded 3
            agent_cell_radius=agent_cell_radius,
        )
        self.inst_goal_id = None
        self.inst_goal_found = False

        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
            close_frontier_radius=10.0,  #! myTODO: Hardcoded 5
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            panorama_start_steps = 0

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
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
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories+1, device=self.device
        )

        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        self.image_matching_function = self.matching.match_image_to_image
        self.matching_fn = {
            "imagenav": self.image_matching_function,
            "languagenav": self.matching.match_language_to_image,
            "objectnav": None,
        }

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        obs_match_confidences=None,
        obs_match_instance_ids=None,
        mem_match_confidences=None,
        mem_match_instance_ids=None,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
        score_thresh: float = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
                the main inference function of the agent that lets it interact with
                vectorized environments.
        s
                This function assumes that the agent has been initialized.

                Args:
                    obs: current frame containing (RGB, depth, segmentation) of shape
                     (3 + 1 + num_sem_categories, frame_height, frame_width)
                    pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
                     of shape (3)
                    object_goal_category: semantic category of small object goals
                    camera_pose: camera extrinsic pose of shape (4, 4)

                Returns:
                    planner_inputs: list of num_environments planner inputs dicts containing
                        obstacle_map: (M, M) binary np.ndarray local obstacle map
                         prediction
                        sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                         and local map boundaries planning window (gx1, gx2, gy1, gy2)
                        goal_map: (M, M) binary np.ndarray denoting goal location
                    vis_inputs: list of num_environments visualization info dicts containing
                        explored_map: (M, M) binary np.ndarray local explored map
                         prediction
                        semantic_map: (M, M) np.ndarray containing local semantic map
                         predictions
        """
        # * before module call obs.shape is [380+3+1+num_instances]

        last_step_local_id_to_global_id_map = (
            self.instance_memory.local_id_to_global_id_map.copy()
        )
        #! myTODO: Clean this up
        self.semantic_map_module.vis_dir = self.planner.vis_dir
        self.semantic_map_module.timestep = self.get_subtask_timestep() + 1
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
            obstacle_locations=obstacle_locations,
            free_locations=free_locations,
            blacklist_target=blacklist_target,
        )

        if self.inst_goal_found:
            logger.info(f"Already found instance goal, not searching anymore.")
        else:
            logger.debug(
                f"candidate matches in memory: {len(mem_match_confidences)}, candidate matches in observation: {len(obs_match_confidences)}"
            )
            if len(mem_match_confidences) > 0 or len(obs_match_confidences) > 0:
                (
                    self.inst_goal_found,
                    self.inst_goal_id,
                ) = self.matching.get_best_inst_goal(
                    obs_match_confidences,
                    obs_match_instance_ids,
                    last_step_local_id_to_global_id_map,
                    mem_match_confidences=mem_match_confidences,
                    mem_match_instance_ids=mem_match_instance_ids,
                    score_thresh=score_thresh,
                )

        #! myTODO: Make this simpler.
        self.semantic_map.vis_dir = self.planner.vis_dir

        self.total_timesteps = self.total_timesteps + 1
        self.sub_task_timesteps[self.current_task_idx] += 1
    
    def get_subtask_timestep(self) -> int:
        return self.sub_task_timesteps[self.current_task_idx]

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.inst_goal_found = False
        self.inst_goal_id = None

    def reset(self):
        """Initialize agent state. Reset is at the beginning of a new episode (not each task)."""
        self.total_timesteps = 0
        self.sub_task_timesteps = [0] * self.max_num_sub_task_episodes
        self.last_poses = np.zeros(3)
        self.semantic_map.init_map_and_pose()
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.navigate_to_best = False

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()

        self.reset_sub_episode()
        self.planner.reset()
        self.inst_goal_found = False
        self.inst_goal_id = None

    def score_thresh(self, task_type):
        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def act(
        self, obs: Observations, stop=False
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        is_local = True
        logger.info(
            f"---------------- Subtask step {self.get_subtask_timestep() + 1} ----------------"
        )
        logger.debug(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
        current_task = obs.task_observations["tasks"][self.current_task_idx]

        (
            obs_preprocessed,
            pose_delta,
            obs_match_confidences,
            obs_match_instance_ids,
            mem_match_confidences,
            mem_match_instance_ids,
        ) = self._preprocess_obs(obs, current_task["type"])

        self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            reject_visited_targets=self.reject_visited_targets,
            obs_match_confidences=obs_match_confidences,
            obs_match_instance_ids=obs_match_instance_ids,
            mem_match_confidences=mem_match_confidences,
            mem_match_instance_ids=mem_match_instance_ids,
            score_thresh=self.score_thresh(current_task["type"]),
        )

        if (
            self.get_subtask_timestep() 
            >= self.max_steps[self.current_task_idx]
        ) or stop:
            logger.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            action = DiscreteNavigationAction.STOP
            vis_inputs = {}
        else:
            action, vis_inputs = self.get_best_action(current_task)

        if self.visualize:
            is_local = vis_inputs.get("is_local", True)
            vis_inputs = {
                "inst_goal_id": self.inst_goal_id,
                "timestep": self.get_subtask_timestep(),
                "total_timesteps": self.total_timesteps,
                "explored_map": self.semantic_map.get_explored_map(is_local),
                "obstacle_map": self.semantic_map.get_obstacle_map(is_local),
                "semantic_map": self.semantic_map.get_semantic_map(is_local),
                "frontier_map": self.semantic_map.get_frontier_map(is_local),
                "been_close_map": self.semantic_map.get_been_close_map(is_local),
                "global_pose": self.semantic_map.global_pose,
                "lmb": self.semantic_map.lmb,
                **vis_inputs,
            }

            if current_task["type"] == "imagenav":
                collision = {"is_collision": False}
                info = {
                    **vis_inputs,
                    "rgb_frame": obs.rgb,
                    "semantic_frame": obs.semantic,
                    "last_goal_image": obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                }
                if self.imagenav_visualizer is not None:
                    self.imagenav_visualizer.visualize(**info)
            else:
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                vis_inputs["goal_name"] = goal_text_desc
                vis_inputs["semantic_frame"] = obs.task_observations["semantic_frame"]
                vis_inputs["third_person_image"] = obs.third_person_image
                vis_inputs["instance_memory"] = self.instance_memory

                info = vis_inputs
        else:
            info = None

        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.reset_sub_episode()
                self.current_task_idx += 1
                self.navigate_to_best = False
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        current_task = obs.task_observations["tasks"][self.current_task_idx]

        (
            obs_match_confidences,
            obs_match_instance_ids,
            mem_match_confidences,
            mem_match_instance_ids,
        ) = (
            [],
            [],
            [],
            [],
        )

        if not self.inst_goal_found:
            obs_match_confidences, obs_match_instance_ids = (
                self._match_against_current_frame(task_type, current_task, obs)
            )

        # * Semantics becomes (W,H,NumClasses) which NumClasses is read from the config files, and is 380. Note that because I am using less classes (52 in all_ovon_categires) most of these layers are zero and actually useless.
        # * Maybe I should change the config. But nevertheles, this works even with 380.
        semantic = self.one_hot_encoding[torch.from_numpy(obs.semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            # * Why using instance_map which are the raw semantics? To differentiate between objects with diff raw semantics but same category in our ovon classes.
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            mem_match_instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx
                for idx, instance_id in enumerate(mem_match_instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(mem_match_instance_ids), device=self.device)[
                instances
            ]

            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses), device=rgb.device
        )
        self.last_poses = curr_pose

        assert obs.camera_pose is None

        # Match a goal against every instance in memory the moment we get it
        # or when the map just got fully explored
        if self.get_subtask_timestep() == 0:
            mem_match_confidences, mem_match_instance_ids = self._match_against_memory(
                current_task
            )

        # * preprocessed obs shape is (1, 3+1+num_sem_classes+num_instances, H, W)
        return (
            obs_preprocessed,
            pose_delta,
            obs_match_confidences,
            obs_match_instance_ids,
            mem_match_confidences,
            mem_match_instance_ids,
        )

    def _match_against_current_frame(self, task_type, current_task, obs):
        image_goal = None
        language_goal = None

        if task_type == "imagenav":
            if self.goal_image is None:
                img_goal = obs.task_observations["tasks"][self.current_task_idx][
                    "image"
                ]
                self.goal_image, self.goal_image_keypoints = (
                    self.matching.get_goal_image_keypoints(img_goal)
                )
                # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)
            image_goal = self.goal_image

        elif task_type == "languagenav":
            language_goal = current_task["description"]

        confidences, frame_matches_local_instance_ids = (
            self.matching.get_matches_against_current_frame(
                self.matching_fn[task_type],
                self.total_timesteps,
                image_goal=image_goal,
                goal_image_keypoints=self.goal_image_keypoints,
                language_goal=language_goal,
                categories=[current_task["semantic_id"]],
                use_full_image=False,
                global_pose=self.semantic_map.global_pose,
            )
        )

        return confidences, frame_matches_local_instance_ids

    def _match_against_memory(self, current_task: Dict):
        task_type = current_task["type"]
        logger.info("--------Matching against memory!--------")
        image_goal = None
        language_goal = None
        goal_image_keypoints = None
        if task_type == "languagenav":
            language_goal = current_task["description"]
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
            categories=[current_task["semantic_id"]],
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

    def get_best_action(self, current_task):
        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            current_task["semantic_id"],
        )

        if not action is None:
            return action, vis_input

        logger.info("No reachable goal.")

        if self.navigate_to_best:
            logger.info("Already tried the best match. Stopping")
            return DiscreteNavigationAction.STOP, {}
        self.navigate_to_best = True
        logger.info("Forcing a match against memory")

        mem_match_confidences, mem_match_instance_ids = self._match_against_memory(
            current_task
        )
        if not len(mem_match_confidences) > 0:
            logger.info("No match found in memory. Stopping")
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
            logger.info("Best match is the same as the previous one. Stopping")
            return DiscreteNavigationAction.STOP, {}

        action, vis_input = self.planner.plan(
            self.inst_goal_found,
            self.inst_goal_id,
            self.get_subtask_timestep(),
            self.total_timesteps,
            current_task["semantic_id"],
            fallback_to_frontier=False,
            postfix="_last_shot",
        )

        if action is None:
            logger.info("Fully explored and no path to our best match. Stopping")
            return DiscreteNavigationAction.STOP, {}

        logger.info("Found a path to the last shot goal. Navigating to it.")
        return action, vis_input
