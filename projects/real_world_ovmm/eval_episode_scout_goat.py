#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Real-world GOAT evaluation script for Scout robot
Supports multi-task episodes with ObjectNav, ImageNav, and LanguageNav
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import rospy
import numpy as np
from tqdm import tqdm

# Add home_robot to path if needed
home_robot_path = Path(__file__).resolve().parent.parent.parent / "src/home_robot"
if home_robot_path.exists():
    sys.path.insert(0, str(home_robot_path))

home_robot_hw_path = Path(__file__).resolve().parent.parent.parent / "src/home_robot_hw"
if home_robot_hw_path.exists():
    sys.path.insert(0, str(home_robot_hw_path))

from omegaconf import DictConfig, OmegaConf
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.utils.logger import get_logger

# Import our GOAT components
from home_robot_hw.env.scout_goat_env import ScoutGoatEnv
from home_robot.agent.goat_agent.goat_agent_scout import ScoutGoatAgent

logger = get_logger()


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file"""
    config = OmegaConf.load(config_path)
    
    # Set defaults if not specified
    if not hasattr(config, 'EVAL_SETTINGS'):
        config.EVAL_SETTINGS = DictConfig({
            'control_frequency': 1.0,
            'wait_for_action': True,
            'reset_nav': False,
            'episode_timeout_minutes': 15,
            'task_timeout_steps': 500,
            'save_trajectory': True,
            'save_semantic_maps': True,
            'log_verbose': True,
        })
    
    return config


def save_task_results(
    results: dict,
    env: ScoutGoatEnv,
    results_dir: Path,
    task_idx: int,
    task_timesteps: int,
    task_metrics: dict,
):
    """Save results for completed task"""
    task_key = f"{env.scene_id}_{env.episode_id}_task_{task_idx}"
    
    task_info = env.current_episode["tasks"][task_idx] if task_idx < len(env.current_episode["tasks"]) else {}
    
    results[task_key] = {
        "episode_id": env.episode_id,
        "task_idx": task_idx,
        "task_type": task_info.get("type", "unknown"),
        "timesteps": task_timesteps,
        "metrics": task_metrics,
    }
    
    # Save per-task results
    try:
        with open(results_dir / "per_task_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Task {task_idx + 1} results saved")
    except Exception as e:
        logger.error(f"Failed to save task results: {e}")


def save_episode_results(
    results: dict,
    env: ScoutGoatEnv,
    results_dir: Path,
    episode_timesteps: int,
    all_task_metrics: list,
    agent,
):
    """Save results for completed episode"""
    episode_key = f"{env.scene_id}_{env.episode_id}"
    
    episode_results = {
        "episode_id": env.episode_id,
        "total_timesteps": episode_timesteps,
        "num_tasks": len(env.current_episode["tasks"]),
        "tasks_completed": len(all_task_metrics),
        "sub_task_timesteps": agent.sub_task_timesteps[:len(all_task_metrics)],
        "task_metrics": all_task_metrics,
        "tasks": [
            {
                "type": task["type"],
                "category": task.get("category", ""),
                "description": task.get("description", "")
            }
            for task in env.current_episode["tasks"]
        ]
    }
    
    # Compute aggregate statistics
    if len(all_task_metrics) > 0 and len(agent.sub_task_timesteps) > 0:
        valid_timesteps = agent.sub_task_timesteps[:len(all_task_metrics)]
        episode_results["avg_timesteps_per_task"] = float(np.mean(valid_timesteps))
    
    results[episode_key] = episode_results
    
    # Save per-episode results
    try:
        with open(results_dir / "per_episode_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Episode {env.episode_id} results saved")
    except Exception as e:
        logger.error(f"Failed to save episode results: {e}")


def compute_cumulative_stats(results: dict, results_dir: Path):
    """Compute and save cumulative statistics across all episodes"""
    if not results:
        return
        
    all_task_timesteps = []
    task_types = {"objectnav": [], "imagenav": [], "languagenav": []}
    
    for key, data in results.items():
        # Skip task-level results, only process episode-level
        if "_task_" in key:
            continue
            
        if "task_metrics" in data and "tasks" in data:
            for task_idx, metrics in enumerate(data["task_metrics"]):
                timesteps = data["sub_task_timesteps"][task_idx] if task_idx < len(data["sub_task_timesteps"]) else 0
                all_task_timesteps.append(timesteps)
                
                # Group by task type
                if task_idx < len(data["tasks"]):
                    task_type = data["tasks"][task_idx].get("type", "unknown")
                    if task_type in task_types:
                        task_types[task_type].append(timesteps)
    
    stats = {
        "total_episodes": len([k for k in results.keys() if "_task_" not in k]),
        "total_tasks": len(all_task_timesteps),
    }
    
    if all_task_timesteps:
        stats["avg_timesteps_per_task"] = float(np.mean(all_task_timesteps))
        stats["median_timesteps_per_task"] = float(np.median(all_task_timesteps))
    
    # Per-task-type statistics
    for task_type, timesteps_list in task_types.items():
        if len(timesteps_list) > 0:
            stats[f"{task_type}_count"] = len(timesteps_list)
            stats[f"{task_type}_avg_timesteps"] = float(np.mean(timesteps_list))
            stats[f"{task_type}_median_timesteps"] = float(np.median(timesteps_list))
    
    try:
        with open(results_dir / "cumulative_metrics.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        logger.info("\nCumulative statistics:")
        logger.info(json.dumps(stats, indent=2))
    except Exception as e:
        logger.error(f"Failed to save cumulative stats: {e}")


@click.command()
@click.option(
    "--config",
    default="projects/real_world_ovmm/configs/agent/eval.yaml",
    help="Path to configuration YAML file",
)
@click.option(
    "--task-config",
    default="projects/real_world_ovmm/configs/example_tasks.json",
    help="Path to task definitions JSON file",
)
@click.option(
    "--num-episodes",
    default=None,
    type=int,
    help="Number of episodes to run (None = all episodes in task config)",
)
@click.option(
    "--reset-nav",
    default=False,
    is_flag=True,
    help="Reset robot to origin before starting",
)
@click.option(
    "--control-frequency",
    default=None,
    type=float,
    help="Control loop frequency in Hz (overrides config)",
)
@click.option(
    "--visualize",
    default=None,
    type=bool,
    help="Enable visualization (overrides config)",
)
def main(
    config: str,
    task_config: str,
    num_episodes: Optional[int],
    reset_nav: bool,
    control_frequency: Optional[float],
    visualize: Optional[bool],
):
    """
    Run GOAT evaluation on Scout robot
    
    This script runs multi-task navigation episodes on a physical Scout robot.
    Each episode contains multiple sequential tasks (ObjectNav, ImageNav, or LanguageNav).
    """
    print("=" * 80)
    print("SCOUT ROBOT - GOAT MULTI-TASK NAVIGATION EVALUATION")
    print("=" * 80)
    
    # Initialize ROS
    print("\n[1/6] Initializing ROS node...")
    rospy.init_node("eval_episode_scout_goat")
    
    # Load configuration
    print("[2/6] Loading configuration...")
    cfg = load_config(config)
    
    # Override config with command-line arguments
    if control_frequency is not None:
        cfg.EVAL_SETTINGS.control_frequency = control_frequency
    if visualize is not None:
        cfg.VISUALIZE = visualize
        
    print(f"  Config: {config}")
    print(f"  Task config: {task_config}")
    print(f"  Control frequency: {cfg.EVAL_SETTINGS.control_frequency} Hz")
    print(f"  Visualization: {cfg.VISUALIZE}")
    
    # Create environment
    print("\n[3/6] Creating environment...")
    env = ScoutGoatEnv(
        config=cfg,
        task_config_file=task_config,
        forward_step=cfg.ENVIRONMENT.forward,
        rotate_step=cfg.ENVIRONMENT.turn_angle,
    )
    
    print(f"  Loaded {len(env.episodes)} episodes")
    print(f"  Vocabulary: {env.vocabulary}")
    
    # Create agent
    print("\n[4/6] Creating agent...")
    agent = ScoutGoatAgent(
        config=cfg,
        semantic_category_mapping=env.semantic_category_mapping,
        device_id=0,
    )
    print("  Agent initialized with semantic mapping")
    
    # Setup results directory
    results_dir = Path(cfg.DUMP_LOCATION) / "results" / cfg.EXP_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[5/6] Results directory: {results_dir}")
    
    # Reset robot to origin if requested
    if reset_nav:
        print("\n[*] Resetting robot to origin [0, 0, 0]...")
        robot = env.get_robot()
        robot.nav.navigate_to([0, 0, 0])
        time.sleep(2.0)
    
    # Setup control rate
    rate = rospy.Rate(cfg.EVAL_SETTINGS.control_frequency)
    
    # Main evaluation loop
    print("\n[6/6] Starting evaluation...")
    print("=" * 80)
    
    results = {}
    episode_count = 0
    max_episodes = num_episodes if num_episodes is not None else len(env.episodes)
    
    while episode_count < max_episodes and not rospy.is_shutdown():
        # Reset environment for new episode
        env.reset()
        episode_start_time = time.time()
        
        # Reset agent
        agent.reset(
            scene_id=env.scene_id,
            episode_id=env.episode_id,
            current_task_idx=0,
        )
        
        print(f"\n{'='*80}")
        print(f"EPISODE {env.episode_id + 1}/{max_episodes}")
        print(f"{'='*80}")
        
        episode_timesteps = 0
        all_task_metrics = []
        task_start_step = 0
        
        # Episode loop
        # Episode loop
        while not env.episode_over and not rospy.is_shutdown():
            step_start_time = time.time()
            
            # Check episode timeout
            elapsed_minutes = (time.time() - episode_start_time) / 60
            if elapsed_minutes > cfg.EVAL_SETTINGS.episode_timeout_minutes:
                print(f"\n⚠️  Episode timeout ({cfg.EVAL_SETTINGS.episode_timeout_minutes} minutes)")
                break
            
            # Get observation
            obs = env.get_observation()
            
            # CRITICAL: Update agent state with observations
            agent.update_state(obs)
            
            # Get action from agent
            action, info, stuck = agent.act()
            # After: action, info, stuck = agent.act()

            
            # =========================================================
            # GOAL DETECTION FEEDBACK
            # =========================================================
            goal_detected = False
            inst_goal_id = None
            
            # Check various goal detection flags
            if info:
                goal_detected = info.get('found_goal', False) or info.get('inst_goal_found', False)
                inst_goal_id = info.get('inst_goal_id')
            
            # Also check agent attributes directly
            if hasattr(agent, 'inst_goal_found') and agent.inst_goal_found:
                goal_detected = True
            if hasattr(agent, 'inst_goal_id') and agent.inst_goal_id is not None:
                inst_goal_id = agent.inst_goal_id
            
            # Check if in panorama phase
            is_panorama = False
            if hasattr(agent, 'timesteps') and hasattr(agent, 'panorama_start_steps'):
                current_step = agent.timesteps[0] if hasattr(agent.timesteps, '__getitem__') else agent.timesteps
                is_panorama = current_step <= getattr(agent, 'panorama_start_steps', 0)
            elif hasattr(agent, 'total_timesteps'):
                # GoatAgent uses total_timesteps
                panorama_steps = cfg.AGENT.get('panorama_start', 12)
                is_panorama = agent.total_timesteps <= panorama_steps
            
            # =========================================================
            # Handle stuck/timeout
            if stuck:
                print("\n⚠️  Agent stuck or task timeout - forcing STOP")
                action = DiscreteNavigationAction.STOP
            
            # Display current task info
            current_task = obs.task_observations["tasks"][env.current_task_idx]
            task_type = current_task["type"].upper()
            task_desc = current_task.get("category", current_task.get("description", ""))
            
            # =========================================================
            # ENHANCED LOGGING
            # =========================================================
            print(f"\n{'='*60}")
            print(f"  🔎 Goal semantic_id={current_task.get('semantic_id')} | inst_goal_found={getattr(agent, 'inst_goal_found', '?')} | inst_goal_id={getattr(agent, 'inst_goal_id', '?')}")
            print(f"Step {episode_timesteps + 1} | Task {env.current_task_idx + 1}/{len(obs.task_observations['tasks'])}")
            print(f"  Type: {task_type} | Target: {task_desc}")
            print(f"  Action: {action.name}")

            
            # Goal detection status
            if goal_detected:
                print(f"  🎯 GOAL DETECTED! Instance ID: {inst_goal_id}")
            else:
                print(f"  🔍 Goal not yet detected")
            
            # Panorama status
            if is_panorama:
                print(f"  📷 Panorama scan in progress...")
            
            # Show semantic detections from this frame
            if hasattr(obs, 'semantic') and obs.semantic is not None:
                unique_ids = np.unique(obs.semantic)
                detected_categories = []
                for sem_id in unique_ids:
                    if sem_id > 0:  # Skip background
                        # Try to get category name
                        cat_name = f"id_{sem_id}"
                        if hasattr(env, 'semantic_category_mapping'):
                            cat_name = env.semantic_category_mapping.get_category_name(int(sem_id))
                        pixel_count = (obs.semantic == sem_id).sum()
                        if pixel_count > 100:  # Only show significant detections
                            detected_categories.append(f"{cat_name}({pixel_count}px)")
                
                if detected_categories:
                    print(f"  👁️  Detected: {', '.join(detected_categories[:5])}")  # Show top 5
                else:
                    print(f"  👁️  No objects detected in frame")
            
            # Show exploration info if available
            if info:
                if 'closest_goal_pt' in info and info['closest_goal_pt'] is not None:
                    print(f"  📍 Goal point: {info['closest_goal_pt']}")
                if 'short_term_goal' in info and info['short_term_goal'] is not None:
                    print(f"  🎯 Short-term goal: {info['short_term_goal']}")
            
            print(f"{'='*60}")
            
            # Execute action
            done = env.apply_action(action, info=info, prev_obs=obs)
            
            episode_timesteps += 1
            
            # Task completed
            if action == DiscreteNavigationAction.STOP:
                print(f"  ⚠️  STOP triggered!")
                print(f"      - Goal detected: {goal_detected}")
                print(f"      - Stuck: {stuck}")
                print(f"      - Total timesteps: {episode_timesteps}")
                if hasattr(agent, 'sub_task_timesteps'):
                    print(f"      - Sub-task timesteps: {agent.sub_task_timesteps}")
                task_timesteps = agent.sub_task_timesteps[env.current_task_idx - 1]
                task_metrics = {
                    "task_type": current_task["type"],
                    "timesteps": task_timesteps,
                    "success": not stuck,  # Simple success metric
                }
                
                all_task_metrics.append(task_metrics)
                
                print(f"\n✓ Task {env.current_task_idx} complete in {task_timesteps} steps")
                
                # Save task results
                save_task_results(
                    results,
                    env,
                    results_dir,
                    env.current_task_idx - 1,
                    task_timesteps,
                    task_metrics,
                )
                
                # Reset visualization for next task if episode continues
                if not env.episode_over:
                    env.reset_visualization()
                    agent._reset_vis_dir(env.scene_id, env.episode_id, env.current_task_idx)
                    task_start_step = episode_timesteps
            
            # Rate limiting
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                print("\n⚠️  ROS interrupted")
                break
        
        # Episode complete
        episode_duration = time.time() - episode_start_time
        
        print(f"\n{'='*80}")
        print(f"EPISODE {env.episode_id + 1} COMPLETE")
        print(f"{'='*80}")
        print(f"  Total steps: {episode_timesteps}")
        print(f"  Duration: {episode_duration:.1f}s ({episode_duration/60:.1f} min)")
        print(f"  Tasks completed: {len(all_task_metrics)}/{len(env.current_episode['tasks'])}")
        print(f"  Avg steps per task: {np.mean(agent.sub_task_timesteps[:len(all_task_metrics)]):.1f}")
        
        # Save episode results
        save_episode_results(
            results,
            env,
            results_dir,
            episode_timesteps,
            all_task_metrics,
            agent,
        )
        
        # Move to next episode
        episode_count += 1
        if episode_count < max_episodes:
            has_next = env.next_episode()
            if not has_next:
                print("\n✓ All episodes in config completed")
                break
        else:
            print(f"\n✓ Completed {max_episodes} episodes")
            break
    
    # Compute final statistics
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    compute_cumulative_stats(results, results_dir)
    
    print(f"\nResults saved to: {results_dir}")
    print("=" * 80)


def cleanup_ros():
    """Gracefully shutdown ROS to prevent 'closed topic' errors."""
    try:
        # Give time for final messages
        rospy.sleep(0.5)
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ros()
        print("\n✅ ROS shutdown complete. Exiting.")