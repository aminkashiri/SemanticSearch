# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


<<<<<<< HEAD
=======
from datetime import datetime
from enum import IntEnum, auto
>>>>>>> atharva/goat-sim
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
<<<<<<< HEAD
import rospy

from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations


class ScoutAgent(GoatAgent):
    """
    Simplified Scout agent for pure object navigation using GoatAgent.
    No manipulation/grasping - just navigation to objects.
    """

    def __init__(self, config, semantic_category_mapping, device_id: int = 0, agent_id=None):
        # Initialize GoatAgent with required parameters
        super().__init__(config, semantic_category_mapping, agent_id=agent_id, device_id=device_id)
        
        # Check if ROS node is initialized
        if not rospy.core.is_initialized():
            rospy.init_node("scout_agent_node", anonymous=True)
        
        self.config = config
        print("[SCOUT AGENT] Initialized for pure object navigation (no manipulation)")

    def reset(self, scene_id, episode_id, current_task_idx=0):
        """Initialize agent state for new episode."""
        # Call parent reset with proper parameters
        super().reset(scene_id, episode_id, current_task_idx)
        print(f"[SCOUT AGENT] Reset for scene: {scene_id}, episode: {episode_id}")

    def act(self, other_agents=None) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """
        Act using GoatAgent's navigation.
        Note: GoatAgent.act() does NOT take obs as parameter - state is already updated via update_state()
        Returns: (action, info, stuck)
        """
        # Get action from GoatAgent (no obs parameter!)
        action, info, stuck = super().act(other_agents=other_agents)
        
        if self.verbose:
            goal_found = info.get('found_goal', False) or getattr(self, 'inst_goal_found', False)
            print(f"[SCOUT AGENT] Step {self.get_subtask_timestep()}: "
                  f"Action={action}, GoalFound={goal_found}, Stuck={stuck}")
        
        return action, info, stuck
=======
# Import ROS for the node check fix
import rospy

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ovmm_perception import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
# from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations


class Skill(IntEnum):
    """
    Enumerates the different skills the agent can perform.
    Each skill represents a distinct phase of the navigation task.
    """

    NAV_TO_OBJ = auto()  # Navigate to the object to be picked up
    GAZE_AT_OBJ = auto()  # Gaze at the object to get a better view
    STOP = auto()  # Stop the episode once the goal is reached


class SemanticVocab(IntEnum):
    """
    Enumerates the different types of semantic vocabularies for perception.
    """

    FULL = auto()  # Full vocabulary, likely includes all known objects/receptacles
    SIMPLE = auto()  # Simple vocabulary, restricted to the objects relevant for the current task
    ALL = auto()  # All vocabulary, includes all possible categories


def get_skill_as_one_hot_dict(curr_skill: Skill):
    """
    Creates a one-hot encoded dictionary representing the current skill.
    This is useful for providing skill information to an RL agent.
    """
    skill_dict = {f"is_curr_skill_{skill.name}": 0 for skill in Skill}
    skill_dict[f"is_curr_skill_{Skill(curr_skill).name}"] = 1
    return skill_dict


class ScoutAgent(ObjectNavAgent):
    """
    A simplified agent for the Scout robot, focused on open-vocabulary navigation
    and object-gazing, without manipulation capabilities.
    """

    def __init__(self, config, device_id: int = 0):
        # NOTE: ObjectNavAgent.__init__ calls super() which is ObjectNavAgent. This seems correct.
        super().__init__(config, device_id=device_id)
        
        # Check if ROS node is initialized before using it indirectly
        # This fixes an early ROSException if init_node was called multiple times
        if not rospy.core.is_initialized():
            rospy.init_node("ovmm_agent_node", anonymous=True)

        # Initialize state variables for the agent's state machine
        self.states = None
        self.gaze_at_obj_start_step = None
        self.gaze_agent = None
        self.nav_to_obj_agent = None
        self.semantic_sensor = None

        self.skip_skills = config.AGENT.skip_skills
        if config.GROUND_TRUTH_SEMANTICS == 0:
            # Initialize the perception module for semantic segmentation
            self.semantic_sensor = OvmmPerception(config, device_id, self.verbose)
            # Read the category map for object and receptacle names
            self.obj_name_to_id, self.rec_name_to_id = read_category_map_file(
                config.ENVIRONMENT.category_map_file
            )
        # Initialize RL-based gaze agent if configured
        if config.AGENT.SKILLS.GAZE_OBJ.type == "rl" and not self.skip_skills.gaze_at_obj:
            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.GAZE_OBJ,
                device_id=device_id,
            )
        # Initialize RL-based navigation to object agent if configured
        if (
            config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
            and not self.skip_skills.nav_to_obj
        ):
            self.nav_to_obj_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.NAV_TO_OBJ,
                device_id=device_id,
            )
        self.config = config

    def _get_info(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        use_detic_viz = self.config.ENVIRONMENT.use_detic_viz

        if self.config.GROUND_TRUTH_SEMANTICS == 1 or use_detic_viz:
            semantic_category_mapping = None
        elif self.semantic_sensor.current_vocabulary_id == SemanticVocab.SIMPLE:
            # A simple vocabulary for Scout would just be the object itself
            # FIX: Use 'goal_name' from obs.task_observations
            semantic_category_mapping = {
                obs.task_observations["goal_name"]: 1
            }
        else:
            semantic_category_mapping = self.semantic_sensor.current_vocabulary

        if use_detic_viz:
            semantic_frame = obs.task_observations["semantic_frame"]
        else:
            semantic_frame = np.concatenate(
                [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
            ).astype(np.uint8)

        # Create the info dictionary for the agent
        info = {
            "semantic_frame": semantic_frame,
            "semantic_category_mapping": semantic_category_mapping,
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
            "curr_skill": Skill(self.states[0].item()).name,
            "skill_done": "",
        }
        info = {**info, **get_skill_as_one_hot_dict(self.states[0].item())}
        return info

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()

    def reset_vectorized(self):
        """Initialize agent state across all environments."""
        super().reset_vectorized()
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.gaze_at_obj_start_step = torch.tensor([0] * self.num_environments)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.gaze_at_obj_start_step[e] = 0
        super().reset_vectorized_for_env(e)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized_for_env(e)

    def _init_episode(self, obs: Observations):
        """
        This method is called at the first timestep of every episode before any action is taken.
        """
        if self.verbose:
            print("Initializing episode...")
        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            self._update_semantic_vocabs(obs)
            if (
                self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
                and not self.skip_skills.nav_to_obj
            ):
                self._set_semantic_vocab(SemanticVocab.FULL, force_set=True)
            else:
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=True)

    def _switch_to_next_skill(
        self, e: int, next_skill: Skill, info: Dict[str, Any]
    ) -> DiscreteNavigationAction:
        """Switch to the next skill for environment `e`."""
        action = None
        if next_skill == Skill.NAV_TO_OBJ:
            pass
        elif next_skill == Skill.GAZE_AT_OBJ:
            self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=False)
            self.gaze_at_obj_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.STOP:
            # The Scout agent stops when the goal is reached
            action = DiscreteNavigationAction.STOP
        self.states[e] = next_skill
        return action

    def _update_semantic_vocabs(
        self, obs: Observations, update_full_vocabulary: bool = True
    ):
        """
        Sets vocabularies for semantic sensor at the start of episode.
        """
        # FIX: Changed "object_name" to "goal_name" to resolve KeyError
        obj_id_to_name = {0: obs.task_observations["goal_name"]} 
        
        # For Scout, the simple vocab only needs the object
        simple_vocab = build_vocab_from_category_map(obj_id_to_name, {})
        self.semantic_sensor.update_vocabulary_list(simple_vocab, SemanticVocab.SIMPLE)

        if update_full_vocabulary:
            # Full vocabulary contains the object and all receptacles, even if not used
            full_vocab = build_vocab_from_category_map(
                obj_id_to_name, self.rec_name_to_id
            )
            self.semantic_sensor.update_vocabulary_list(full_vocab, SemanticVocab.FULL)

        all_vocab = build_vocab_from_category_map(
            self.obj_name_to_id, self.rec_name_to_id
        )
        self.semantic_sensor.update_vocabulary_list(all_vocab, SemanticVocab.ALL)

    def _set_semantic_vocab(self, vocab_id: SemanticVocab, force_set: bool):
        """
        Set active vocabulary for semantic sensor to use to the given ID.
        """
        if self.config.GROUND_TRUTH_SEMANTICS == 0 and (
            force_set or self.semantic_sensor.current_vocabulary_id != vocab_id
        ):
            self.semantic_sensor.set_vocabulary(vocab_id)

    def _heuristic_nav(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, bool]:
        """Heuristic nav to object skill execution."""
        
        # FIXED: Properly unpack all 4 return values from ObjectNavAgent.act()
        # ObjectNavAgent.act() returns: (action, info, agent_state, obs)
        action, planner_info, agent_state, returned_obs = super().act(obs)
        
        info = {**planner_info, **info}
        self.timesteps[0] -= 1
        info["timestep"] = self.timesteps[0]
        terminate = (action == DiscreteNavigationAction.STOP)
        
        return action, info, terminate

    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[Optional[DiscreteNavigationAction], Any, Optional[Skill]]:
        nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        if self.skip_skills.nav_to_obj:
            terminate = True
        elif nav_to_obj_type == "heuristic":
            if self.verbose:
                print("[SCOUT AGENT] step heuristic nav policy")
            action, info, terminate = self._heuristic_nav(obs, info)
        elif nav_to_obj_type == "rl":
            action, info, terminate = self.nav_to_obj_agent.act(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
            )
        new_state = None
        if terminate:
            action = None
            new_state = Skill.GAZE_AT_OBJ
        return action, info, new_state

    def _gaze_at_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[Optional[DiscreteNavigationAction], Any, Optional[Skill]]:
        gaze_step = self.timesteps[0] - self.gaze_at_obj_start_step[0]
        if self.skip_skills.gaze_at_obj:
            terminate = True
        elif gaze_step == 0:
            # Scout only navigates, no need to switch modes; just return None to step policy
            # Removed redundant NAV_MODE action for a simple switch
            return None, info, None 
        else:
            action, info, terminate = self.gaze_agent.act(obs, info)
        new_state = None
        if terminate:
            action = None
            new_state = Skill.STOP
        return action, info, new_state

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """State machine - NOTE: Returns 3 values, not 4 like ObjectNavAgent"""
        if self.timesteps[0] == 0:
            self._init_episode(obs)

        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            obs = self.semantic_sensor(obs)
        else:
            obs.task_observations["semantic_frame"] = None
        info = self._get_info(obs)
        self.timesteps[0] += 1

        action = None
        while action is None:
            if self.states[0] == Skill.NAV_TO_OBJ:
                action, info, new_state = self._nav_to_obj(obs, info)
            elif self.states[0] == Skill.GAZE_AT_OBJ:
                action, info, new_state = self._gaze_at_obj(obs, info)
            elif self.states[0] == Skill.STOP:
                action = DiscreteNavigationAction.STOP
                new_state = None
            else:
                raise ValueError

            if new_state:
                info["skill_done"] = info["curr_skill"]
                assert action is None, f"action must be None when switching states, found {action} instead"
                action = self._switch_to_next_skill(0, new_state, info)
        info["curr_skill"] = Skill(self.states[0].item()).name
        if self.verbose:
            print(
                f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
            )
        # ScoutAgent.act() returns only 3 values (no agent_state)
        return action, info, obs
>>>>>>> atharva/goat-sim
