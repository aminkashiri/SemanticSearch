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
        self, config, semantic_category_mapping, agent_id=None, device_id: int = 0
    ):
        super().__init__(config, semantic_category_mapping, agent_id, device_id)
        self.inst_goal_ids = None
        self.tasks_done = None
        # self.tasks_failed = None
        self.active_task = None
        self.others_active_task = None
    
    def act(self, other_agents=None):
        neighbors = self._get_neighbors(other_agents)
        self.communicate(neighbors)
        return super().act(neighbors=neighbors)

    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = super()._preprocess_tasks(tasks_obs)
        if self.inst_goal_ids is None:
            self.inst_goal_ids = [None] * len(tasks)
            self.tasks_done = [False] * len(tasks)
            self.others_active_task = {}
        return tasks

    @torch.no_grad()
    def _search_for_goal(self):
        for i, task in enumerate(self.tasks):
            if self.tasks_done[i]:
                continue

            if self.inst_goal_ids[i] is None or self.total_timesteps % 30 == 0:
                inst_goal_id = self.matching.search_for_goal(
                    task,
                    self.match_memory,
                    self.semantic_map.global_pose,
                )
                if not inst_goal_id is None:
                    self.inst_goal_ids[i] = inst_goal_id
            else:
                self.log.debug(
                    f"Instance {self.inst_goal_ids[i]} already found for goal {i}"
                )

        self.match_memory = False

    def _get_best_action(self, **kwargs):
        if all(self.tasks_done):
            self.log.info("All tasks done. Stopping")
            return (None, DiscreteNavigationAction.STOP), {}

        neighbors = kwargs.get("neighbors", [])
        if self.active_task is None:
            for i in range(len(self.tasks)):
                if (
                    self.tasks_done[i]
                    or self.inst_goal_ids[i] is None
                    or i in self.others_active_task
                ):
                    continue
                self.active_task = i

        if self.active_task is None:
            action, vis_input = self.planner.plan(
                neighbors=neighbors,
            )
            if action is None:
                #! myTODO: change to idle action, because it continuous to run this function every step, even after failing once. Somehow we should stop calcualtions
                self.log.info("No reachable goal/frontier. Stopping")
                return (None, DiscreteNavigationAction.STOP), {} # Stop with None, means idle
            else:
                return (None, action), vis_input
        else:
            action, vis_input = self.planner.plan(
                self.inst_goal_ids[self.active_task],
                self.tasks[self.active_task].goal_semantic_id,
                neighbors=neighbors,
                fallback_to_frontier=False,
            )
            if action is None:
                self.tasks_done[self.active_task] = True
                self.active_task = None
                return self._get_best_action(neighbors=neighbors)
            else:
                return (self.active_task, action), vis_input

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
        self.active_task = None
        self.others_active_task = None

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
                    self.others_active_task[neighbor_task] = 0
                    self.active_task = None
            else:
                self.others_active_task[neighbor_task] = 0


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
