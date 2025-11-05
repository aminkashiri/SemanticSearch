# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
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