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
        # DEBUG: Check map state
        if hasattr(self, 'semantic_map') and hasattr(self.semantic_map, 'local_map'):
            local_map = self.semantic_map.local_map
            if local_map is not None:
                # Channel 0 is usually obstacles, Channel 1 is explored
                obs_channel = local_map[0] if len(local_map.shape) == 3 else local_map
                print(f"🔧 Obstacle map: min={obs_channel.min():.2f}, max={obs_channel.max():.2f}, mean={obs_channel.mean():.2f}")
                print(f"🔧 Obstacle pixels (>0.5): {(obs_channel > 0.5).sum()}")
        
        action, info, stuck = super().act(other_agents=other_agents)
        return action, info, stuck