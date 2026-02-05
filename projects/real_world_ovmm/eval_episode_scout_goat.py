#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Real-world GOAT evaluation script for Scout robot.
Uses unified GoatAgent with ScoutGoatEnv.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

import click
import rospy
import numpy as np
from tqdm import tqdm

home_robot_path = Path(__file__).resolve().parent.parent.parent / "src/home_robot"
if home_robot_path.exists():
    sys.path.insert(0, str(home_robot_path))

home_robot_hw_path = Path(__file__).resolve().parent.parent.parent / "src/home_robot_hw"
if home_robot_hw_path.exists():
    sys.path.insert(0, str(home_robot_hw_path))

from omegaconf import DictConfig, OmegaConf
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.utils.logger import get_logger
from home_robot_hw.env.scout_goat_env import ScoutGoatEnv
from home_robot.agent.goat_agent.goat_agent import GoatAgent

logger = get_logger()


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file"""
    config = OmegaConf.load(config_path)
    
    # Set real-world mode
    config.REAL_WORLD = True
    config.SEQ = 1  # Sequential goals for GOAT
    
    if not hasattr(config, 'EVAL_SETTINGS'):
        config.EVAL_SETTINGS = DictConfig({
            'control_frequency': 1.0,
            'episode_timeout_minutes': 15,
        })
    
    return config


def save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs):
    """Save results - matches simulation format."""
    obs_tasks = []
    for task in obs.task_observations["tasks"]:
        obs_task = {k: v for k, v in task.items() if k != "image"}
        obs_tasks.append(obs_task)

    ep_results = {}
    all_subtask_metrics = [all_subtask_metrics[k] for k in sorted(all_subtask_metrics.keys())]
    
    ep_results["metrics"] = all_subtask_metrics
    ep_results["total_num_steps"] = ep_step
    if agent.seq_goals:
        ep_results["sub_task_timesteps"] = agent.subtask_timesteps[:len(all_subtask_metrics)]
    ep_results["tasks"] = obs_tasks

    if all_subtask_metrics:
        for metric in all_subtask_metrics[0].keys():
            values = [m[metric] for m in all_subtask_metrics]
            ep_results[f"{metric}_mean"] = np.round(np.nanmean(values), 4)
            ep_results[f"{metric}_median"] = np.round(np.nanmedian(values), 4)

    results[f"{env.scene_id}_{env.episode_id}"] = ep_results
    
    with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
        json.dump(results, fp, indent=4)

    # Cumulative stats
    if results:
        stats = {}
        all_metrics = [m for r in results.values() for m in r.get("metrics", [])]
        if all_metrics:
            for metric in all_metrics[0].keys():
                values = [m[metric] for m in all_metrics]
                stats[f"{metric}_mean"] = np.round(np.nanmean(values), 4)
                stats[f"{metric}_median"] = np.round(np.nanmedian(values), 4)
            with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
                json.dump(stats, fp, indent=4)


@click.command()
@click.option("--config", default="projects/real_world_ovmm/configs/agent/eval.yaml")
@click.option("--task-config", default="projects/real_world_ovmm/configs/example_tasks.json")
@click.option("--num-episodes", default=None, type=int)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--control-frequency", default=None, type=float)
def main(
    config: str,
    task_config: str,
    num_episodes: Optional[int],
    reset_nav: bool,
    control_frequency: Optional[float],
):
    """Run GOAT evaluation on Scout robot."""
    print("=" * 80)
    print("SCOUT ROBOT - GOAT EVALUATION (Unified GoatAgent)")
    print("=" * 80)
    
    # Initialize ROS
    rospy.init_node("eval_episode_scout_goat")
    
    # Load config
    cfg = load_config(config)
    if control_frequency is not None:
        cfg.EVAL_SETTINGS.control_frequency = control_frequency
    
    # Create environment (has verbose logging)
    env = ScoutGoatEnv(
        config=cfg,
        task_config_file=task_config,
        forward_step=cfg.ENVIRONMENT.forward,
        rotate_step=cfg.ENVIRONMENT.turn_angle,
    )
    
    # Create unified GoatAgent (REAL_WORLD=True mode)
    agent: GoatAgent = GoatAgent(
        cfg, 
        env.semantic_category_mapping.vocabulary
    )
    
    # Setup results
    results_dir = os.path.join(cfg.DUMP_LOCATION, "results", cfg.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    results_file = os.path.join(results_dir, "per_episode_metrics.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as fp:
            results = json.load(fp)
    
    # Reset robot if requested
    if reset_nav:
        robot = env.get_robot()
        robot.nav.navigate_to([0, 0, 0])
        time.sleep(2.0)
    
    rate = rospy.Rate(cfg.EVAL_SETTINGS.control_frequency)
    max_episodes = num_episodes if num_episodes else len(env.episodes)
    
    # Main loop
    for i in range(max_episodes):
        if i > 0:
            if not env.next_episode():
                break
        
        env.reset()
        
        if f"{env.scene_id}_{env.episode_id}" in results:
            continue
        
        env.reset_vis_dir()
        agent.reset(env.scene_id, env.episode_id)
        
        episode_start = time.time()
        ep_step = 0
        all_subtask_metrics = {}
        obs = None
        
        pbar = tqdm(total=cfg.AGENT.max_steps, file=sys.__stdout__, dynamic_ncols=True)
        old_task_idx = -1
        
        while not env.episode_over and not rospy.is_shutdown():
            # Timeout check
            if (time.time() - episode_start) / 60 > cfg.EVAL_SETTINGS.episode_timeout_minutes:
                logger.warning("Episode timeout")
                break
            
            # Task change logging
            if env.current_task_idx != old_task_idx:
                logger.info(f"Starting task {env.current_task_idx} in {env.scene_id} ep {env.episode_id}")
                old_task_idx = env.current_task_idx
            
            pbar.set_description(f"{env.scene_id}_{env.episode_id}_{env.current_task_idx}")
            ep_step += 1
            
            # Sync timestep
            env.timestep = agent.get_subtask_timestep() + 1
            obs = env.get_observation()
            
            # Agent step
            agent.update_state(obs)
            action, info, stuck = agent.act()
            
            # Handle stuck (same pattern as simulation)
            if stuck and action != DiscreteNavigationAction.STOP:
                action = DiscreteNavigationAction.STOP
                agent.handle_stop({"action": action})
            
            logger.info(f"Action: {action}")
            env.apply_action(action, info)
            pbar.update(1)
            
            # Handle STOP
            if action == DiscreteNavigationAction.STOP:
                env.add_subepisode_metrics(all_subtask_metrics, {"action": action, "action_args": {"task_idx": env.current_task_idx - 1}})
                if not env.episode_over:
                    agent.reset_vis_dir(env.scene_id, env.episode_id, env.current_task_idx)
                    env.reset_vis_dir()
                    pbar.reset()
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        pbar.close()
        logger.info(f"Episode {env.scene_id}_{env.episode_id} over")
        
        if obs is not None:
            save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        try:
            rospy.sleep(0.5)
        except:
            pass