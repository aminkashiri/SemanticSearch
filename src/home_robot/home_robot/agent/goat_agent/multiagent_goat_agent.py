# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .goat_agent import GoatAgent, Task
from typing import Any, Dict, List, Tuple
from home_robot.core.interfaces import DiscreteNavigationAction


class MultiAgentGoatAgent(GoatAgent):
    def __init__(
        self, config, vocabulary, agent_id=None, device_id: int = 0
    ):
        super().__init__(config, vocabulary, agent_id, device_id)
        self.inst_goal_ids = None
        self.tasks_done = None
        self.tasks_failed = None
        self.active_task = None
        self.others_active_task_remaining_time = None
        self.communication_cooldown = 5
        self.active_task_cooldown = 20
    
    def act(self, other_agents=None):
        neighbors = self._get_neighbors(other_agents)
        self.communicate(neighbors)
        return super().act(neighbors=neighbors)

    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = super()._preprocess_tasks(tasks_obs)
        if self.inst_goal_ids is None:
            self.inst_goal_ids = [None] * len(tasks)
            self.tasks_done = [False] * len(tasks)
            self.tasks_failed = [False] * len(tasks)
            self.others_active_task_remaining_time = {}
        return tasks

    @torch.no_grad()
    def _search_for_goal(self):
        for i, task in enumerate(self.tasks):
            if self.tasks_done[i] or self.tasks_failed[i]:
                continue

            if self.inst_goal_ids[i] is None or self.total_timesteps % 10 == 0:
                inst_goal_id = self.matching.search_for_goal(
                    task,
                    self.match_memory,
                    self.semantic_map.global_pose,
                    score_thresh=0 if self.navigate_to_best else None
                )
                if not inst_goal_id is None:
                    self.inst_goal_ids[i] = inst_goal_id
            else:
                self.log.debug(
                    f"Instance {self.inst_goal_ids[i]} already found for goal {i}"
                )

        self.match_memory = False

    def _get_best_action(self, **kwargs):
        if all([self.tasks_done[i] or self.tasks_failed[i] for i in range(len(self.tasks_done))]):
            self.log.info("All tasks done or failed, stopping")
            return (None, DiscreteNavigationAction.STOP), {}

        if self.active_task is None:
            for i in range(len(self.tasks)):
                if (
                    self.tasks_done[i] 
                    or self.tasks_failed[i] 
                    or self.inst_goal_ids[i] is None
                    or self.others_active_task_remaining_time.get(i,0) > 0
                ):
                    continue
                self.active_task = i
                break

        neighbors = kwargs.get("neighbors", [])
        if not self.active_task is None:
            action, vis_input = self.planner.plan(
                self.inst_goal_ids[self.active_task],
                self.tasks[self.active_task].goal_semantic_id,
                neighbors=neighbors,
                fallback_to_frontier=False,
            )
            if action is None:
                self.tasks_failed[self.active_task] = True
                return (self.active_task, DiscreteNavigationAction.STOP), vis_input
            else:
                return (self.active_task, action), vis_input

        # If the code reaches here, active task is None.
        action, vis_input = self.planner.plan(
            neighbors=neighbors,
        )
        if not action is None:
            return (None, action), vis_input

        if self.navigate_to_best:
            self.log.warning("No reachable goal/frontier, and tried all possible goals. Stopping")
            return (None, DiscreteNavigationAction.STOP), {} # Stop with None, means idle
        self.log.warning("Reducing threshold to 0 for all goals from now on.")
        self.navigate_to_best = True
        self.match_memory = True
        self._search_for_goal()

        return self._get_best_action(neighbors=neighbors) 

    def _process_action(self, action):
        return {
            "action": action[1],
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": action[0],
            },
        }

    def reset(self, scene_id, episode_id):
        super().reset(scene_id, episode_id)
        self.inst_goal_ids = None
        self.tasks_done = None
        self.tasks_failed = None
        self.active_task = None
        self.others_active_task_remaining_time = None
        self.last_communication_time = {}

    def reset_vis_dir(self, scene_id, episode_id, current_task_idx=None):
        return super().reset_vis_dir(scene_id, episode_id, None)

    def _get_task_info(self, obs, action, info):
        if not action["action_args"]["task_idx"] is None:
            current_task = obs.task_observations["tasks"][
                action["action_args"]["task_idx"]
            ]
            info["task_type"] = current_task["type"]
            goal_text_desc = {x: y for x, y in current_task.items() if x != "image"}
            info["caption"] = str(goal_text_desc)
            if current_task["type"] == "imagenav":
                info["goal_image"] = current_task["image"]
            else:
                info["third_person_image"] = obs.third_person_image
        else:
            remaining_goals = []
            for i, task in enumerate(obs.task_observations["tasks"]):
                if not self.tasks_done[i]:
                    remaining_goals.append(task["category"])
            info["caption"] = f"Remaining Goals: [{', '.join(remaining_goals)}]"

        info["caption"] += f" | Action: {str(action['action']).split('.')[-1]}"

    def handle_stop(self, action):
        self.reset_for_next_task()
        self.active_task = None
        task_idx = action["action_args"]["task_idx"]
        if not task_idx is None:
            self.tasks_done[task_idx] = True
        else:
            self.log.debug("IDLE. Waiting for other agents to complete their tasks.")

    def communicate(self, neighbors):
        # for key in self.others_active_task_remaining_time:
        for key in list(self.others_active_task_remaining_time.keys()):
            self.others_active_task_remaining_time[key] -= 1
            if self.others_active_task_remaining_time[key] <= 0:
                del self.others_active_task_remaining_time[key]
        for neighbor in neighbors:
            last = self.last_communication_time.get(neighbor.agent_id, -1)
            if self.total_timesteps - last > 0:
                self._communicate_with_single_neighbor(neighbor)

        lmb = self.semantic_map.lmb
        self.semantic_map.local_map[:] = self.semantic_map.global_map[
            :, lmb[0] : lmb[1], lmb[2] : lmb[3]
        ]

    def _communicate_with_single_neighbor(self, neighbor):
        data = self._get_communication_data()
        neighbor_data = neighbor.receive_communication(self.agent_id, data)
        self._merge_communication_data(neighbor_data)
        self.last_communication_time[neighbor_data["agent_id"]] = self.total_timesteps

    def receive_communication(self, sender_id, data):
        self.last_communication_time[sender_id] = self.total_timesteps
        self._merge_communication_data(data)
        return self._get_communication_data()

    def _get_communication_data(self):
        return {
            "map": self.semantic_map.global_map,
            "instances": self.instance_memory.instances,
            "agent_id": self.agent_id,
            "tasks_done": self.tasks_done,
            "active_task": self.active_task,
        }

    def _merge_communication_data(self, data):
        self.semantic_map_module.merge_neighbor_maps(
            data["map"], self.semantic_map.global_map
        )
        # self.semantic_map_module.merge_instance_memory(
        #     data["instances"], self.semantic_map.global_map
        # )
        if self.tasks_done is None:
            self.tasks_done = data["tasks_done"]
        else:
            self.tasks_done = [
                a or b for a, b in zip(self.tasks_done, data["tasks_done"])
            ]

        neighbor_task = data["active_task"]
        if neighbor_task is not None:
            same_task = neighbor_task == self.active_task
            has_priority = data["agent_id"] < self.agent_id

            if same_task:
                if has_priority:
                    self.others_active_task_remaining_time[neighbor_task] = self.active_task_cooldown
                    self.active_task = None
            else:
                self.others_active_task_remaining_time[neighbor_task] = self.active_task_cooldown


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
