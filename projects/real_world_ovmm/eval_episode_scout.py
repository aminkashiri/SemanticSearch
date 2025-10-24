#!/usr/bin/env python
import sys
import os
sys.path.append('/home/atharva/SemanticSearch/src/home_robot')
from datetime import datetime
from typing import Optional, Tuple
import click
import rospy
from home_robot.agent.ovmm_agent.ovmm_agent_scout import ScoutAgent
from home_robot_hw.env.scout_object_nav_env import ScoutObjectNavEnv
from home_robot_hw.utils.config import load_config
import time


@click.command()
# --- ROS and command-line arguments for configuring the robot's behavior ---
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option(
    "--cat-map-file",
    default="projects/real_world_ovmm/configs/example_cat_map.json",
)
@click.option("--max-num-steps", default=200)
@click.option("--visualize-maps", default=False, is_flag=True)
@click.option(
    "--control-frequency",
    default=1.0,
    type=float,
    help="Control loop frequency in Hz (default: 1.0 Hz = 1s per step)",
)
@click.option(
    "--wait-for-action",
    default=True,
    is_flag=True,
    help="Wait for robot to complete action before next step",
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging navigation behavior.",
)
def main(
    reset_nav=False,
    dry_run=False,
    visualize_maps=False,
    cat_map_file=None,
    max_num_steps=200,
    control_frequency=0.25,
    wait_for_action=True,
    **kwargs,
):
    print("- Starting ROS node")
    rospy.init_node("eval_episode_scout_objectnav")
    
    print("- Loading configuration")
    config = load_config(visualize=visualize_maps, **kwargs)
    
    print("- Creating environment")
    env = ScoutObjectNavEnv(
        config=config,
        cat_map_file=cat_map_file,
    )
    
    print("- Creating agent")
    agent = ScoutAgent(config=config)
    
    robot = env.get_robot()
    
    if reset_nav:
        print("- Sending the robot to [0, 0, 0]")
        robot.nav.navigate_to([0, 0, 0])
        # Wait for robot to reach origin
        time.sleep(2.0)
    
    now = datetime.now()
    agent.reset()
    
    if hasattr(agent, "planner"):
        agent.planner.set_vis_dir(
            "real_world", now.strftime("%Y_%m_%d_%H_%M_%S")
        )
    
    env.reset()
    
    # Create rate limiter for control loop
    rate = rospy.Rate(control_frequency)
    print(f"- Control loop frequency: {control_frequency} Hz ({1.0/control_frequency:.3f}s per step)")
    
    t = 0
    start_time = time.time()
    last_found_goal = False
    goal_first_seen_step = None
    
    print(f"- Starting episode at time: {start_time:.2f}s")
    print("=" * 60)
    
    while not env.episode_over and not rospy.is_shutdown():
        step_start_time = time.time()
        t += 1
        
        print(f"\n{'='*60}")
        print(f"STEP {t} | Elapsed: {step_start_time - start_time:.2f}s")
        print(f"{'='*60}")
        
        # Get observation
        obs = env.get_observation()
        
        # Agent decides action
        action_decision_start = time.time()
        action, info, obs = agent.act(obs)
        action_decision_time = time.time() - action_decision_start
        
        # Check for goal detection
        found_goal = info.get('found_goal', False)
        
        # Announce when goal is first visually detected
        if found_goal and not last_found_goal:
            goal_first_seen_step = t
            print("\n" + "🎯" * 30)
            print("║  GOAL OBJECT VISUALLY DETECTED!  ║")
            print("║  First seen at step: {}           ║".format(t))
            print("🎯" * 30 + "\n")
            last_found_goal = True
        
        # Display current detection status
        if found_goal:
            print("👁️  Status: Goal in view (detected)")
        else:
            print("🔍 Status: Exploring (goal not visible)")
        
        print(f"Action: {action} (decision time: {action_decision_time:.3f}s)")
        
        # Execute action on robot
        action_execution_start = time.time()
        done = env.apply_action(action, info=info, prev_obs=obs)
        action_execution_time = time.time() - action_execution_start
        
        print(f"Execution time: {action_execution_time:.3f}s")
        
        # If wait_for_action is True, the apply_action should already block
        # But we still want to maintain consistent control frequency
        step_total_time = time.time() - step_start_time
        print(f"Total step time: {step_total_time:.3f}s")
        
        if done:
            print("\n" + "="*60)
            print("EPISODE COMPLETED")
            print("="*60)
            break
        elif t >= max_num_steps:
            print("\n" + "="*60)
            print("REACHED MAXIMUM STEP LIMIT")
            print("="*60)
            break
        
        # Rate limiting: sleep to maintain consistent frequency
        # This prevents the loop from spinning too fast
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            print("ROS interrupted, stopping...")
            break
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"Total steps: {t}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per step: {total_time/t:.2f}s")
    
    # Goal detection summary
    if goal_first_seen_step is not None:
        print(f"\n🎯 Goal Detection:")
        print(f"   - First detected at step: {goal_first_seen_step}")
        print(f"   - Steps exploring: {goal_first_seen_step}")
        print(f"   - Steps with goal visible: {t - goal_first_seen_step}")
    else:
        print(f"\n❌ Goal never visually detected")
    
    print(f"\nMetrics: {env.get_episode_metrics()}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("Starting Scout Object Navigation Evaluation")
    print("="*60)
    main()
    print("\n" + "="*60)
    print("Done Scout Object Navigation Evaluation")
    print("="*60)