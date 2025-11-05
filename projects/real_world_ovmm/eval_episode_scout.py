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
    default=0.5,
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
@click.option(
    "--agent-id",
    default="scout_0",
    type=str,
    help="Unique identifier for this agent",
)
@click.option(
    "--goal",
    default=None,
    type=str,
    help="Specific goal object to navigate to (e.g., 'cup', 'chair')",
)
def main(
    reset_nav=False,
    dry_run=False,
    visualize_maps=False,
    cat_map_file=None,
    max_num_steps=200,
    control_frequency=0.25,
    wait_for_action=True,
    agent_id="scout_0",
    debug=False,
    goal=None,
    **kwargs,
):
    print("=" * 60)
    print("Starting Scout Object Navigation Evaluation")
    print("(Using GoatAgent-based ScoutAgent)")
    print("=" * 60)
    
    print("\n[INIT] Starting ROS node")
    rospy.init_node("eval_episode_scout_objectnav")
    
    print("[INIT] Loading configuration")
    config = load_config(visualize=visualize_maps, **kwargs)
    
    print("[INIT] Creating environment")
    env = ScoutObjectNavEnv(
        config=config,
        cat_map_file=cat_map_file,
    )
    
    print("[INIT] Getting semantic category mapping")
    semantic_category_mapping = env.get_semantic_category_mapping()
    print(f"       - Categories: {semantic_category_mapping.num_sem_categories}")
    print(f"       - Vocabulary: {semantic_category_mapping.vocabulary}")
    
    print("[INIT] Creating agent (GoatAgent-based)")
    agent = ScoutAgent(
        config=config,
        semantic_category_mapping=semantic_category_mapping,
        device_id=0,
        agent_id=agent_id
    )
    
    robot = env.get_robot()
    
    if reset_nav:
        print("[INIT] Sending the robot to [0, 0, 0]")
        robot.nav.navigate_to([0, 0, 0])
        time.sleep(2.0)
    
    now = datetime.now()
    scene_id = "real_world"
    episode_id = now.strftime("%Y_%m_%d_%H_%M_%S")
    current_task_idx = 0
    
    print(f"[INIT] Resetting agent (scene: {scene_id}, episode: {episode_id}, task: {current_task_idx})")
    agent.reset(scene_id, episode_id, current_task_idx)
    
    if hasattr(agent, "planner"):
        agent.planner.set_vis_dir(scene_id, episode_id)
    
    env.reset()
    
    if goal is not None:
        if not env.set_goal(goal):
            print(f"[ERROR] Goal '{goal}' not in vocabulary!")
            print(f"[ERROR] Available goals: {env.goal_options}")
            return
        print(f"[INFO] Goal set to: {goal}")
    
    rate = rospy.Rate(control_frequency)
    print(f"[INIT] Control loop frequency: {control_frequency} Hz ({1.0/control_frequency:.3f}s per step)")
    
    t = 0
    start_time = time.time()
    last_found_goal = False
    goal_first_seen_step = None
    stuck_counter = 0
    max_stuck_steps = 3
    
    print(f"\n[START] Episode started at time: {start_time:.2f}s")
    print("=" * 60)
    
    while not env.episode_over and not rospy.is_shutdown():
        step_start_time = time.time()
        t += 1
        
        print(f"\n{'='*60}")
        print(f"STEP {t} | Elapsed: {step_start_time - start_time:.2f}s")
        print(f"{'='*60}")
        
        # Get observation
        obs = env.get_observation()
        
        # Update agent state (required for GoatAgent)
        # This must be called BEFORE act()
        try:
            agent.update_state(obs)
            if debug:
                print("[DEBUG] Agent state updated")
        except Exception as e:
            print(f"[ERROR] Error updating state: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Agent decides action
        action_decision_start = time.time()
        try:
            # IMPORTANT: GoatAgent.act() does NOT take obs parameter!
            # State is already updated via update_state() above
            action, info, stuck = agent.act(other_agents=None)
            action_decision_time = time.time() - action_decision_start
            
            # Handle stuck condition like habitat does
            if stuck:
                print("[INFO] Agent is stuck, forcing STOP")
                action = DiscreteNavigationAction.STOP
        except Exception as e:
            print(f"[ERROR] Agent action failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Handle stuck condition
        if stuck:
            stuck_counter += 1
            print(f"⚠️  Robot appears stuck! (count: {stuck_counter}/{max_stuck_steps})")
            if stuck_counter >= max_stuck_steps:
                print("\n" + "🛑"*30)
                print("STOPPING: Robot has been stuck for too long")
                print("🛑"*30 + "\n")
                break
        else:
            stuck_counter = 0
        
        # Check for goal detection
        found_goal = info.get('found_goal', False)
        
        # Also check inst_goal_found from GoatAgent
        if not found_goal and hasattr(agent, 'inst_goal_found'):
            found_goal = agent.inst_goal_found
        
        # Announce when goal is first visually detected
        if found_goal and not last_found_goal:
            goal_first_seen_step = t
            print("\n" + "🎯" * 30)
            print("║  GOAL OBJECT VISUALLY DETECTED!  ║")
            print("║  First seen at step: {:3d}          ║".format(t))
            if hasattr(agent, 'inst_goal_id') and agent.inst_goal_id is not None:
                print("║  Instance ID: {:3d}                ║".format(agent.inst_goal_id))
            print("🎯" * 30 + "\n")
            last_found_goal = True
        
        # Display current detection status
        if found_goal:
            print("👁️  Status: Goal in view (detected)")
        else:
            print("🔍 Status: Exploring (goal not visible)")
        
        # Display current skill if available
        if 'curr_skill' in info:
            print(f"🎭 Skill: {info['curr_skill']}")
        
        print(f"🎬 Action: {action} (decision time: {action_decision_time:.3f}s)")
        
        # Execute action on robot
        action_execution_start = time.time()
        try:
            done = env.apply_action(action, info=info, prev_obs=obs)
            action_execution_time = time.time() - action_execution_start
            print(f"⏱️  Execution time: {action_execution_time:.3f}s")
        except Exception as e:
            print(f"[ERROR] Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        step_total_time = time.time() - step_start_time
        print(f"📊 Total step time: {step_total_time:.3f}s")
        
        if done:
            print("\n" + "="*60)
            print("✅ EPISODE COMPLETED")
            print("="*60)
            break
        elif t >= max_num_steps:
            print("\n" + "="*60)
            print("⏰ REACHED MAXIMUM STEP LIMIT")
            print("="*60)
            break
        
        # Rate limiting: sleep to maintain consistent frequency
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            print("ROS interrupted, stopping...")
            break
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"📝 Total steps: {t}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    if t > 0:
        print(f"📊 Average time per step: {total_time/t:.2f}s")
    
    # Goal detection summary
    if goal_first_seen_step is not None:
        print(f"\n🎯 Goal Detection:")
        print(f"   - First detected at step: {goal_first_seen_step}")
        print(f"   - Steps exploring before detection: {goal_first_seen_step}")
        print(f"   - Steps with goal visible: {t - goal_first_seen_step}")
    else:
        print(f"\n❌ Goal never visually detected")
    
    # Instance memory summary (from GoatAgent)
    if hasattr(agent, 'instance_memory') and agent.instance_memory is not None:
        try:
            num_instances = len(agent.instance_memory.instances) if hasattr(agent.instance_memory, 'instances') else 0
            print(f"\n📦 Instance Memory:")
            print(f"   - Total instances tracked: {num_instances}")
        except:
            pass
    
    print(f"\n📈 Metrics: {env.get_episode_metrics()}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*60)
        print("✅ Done Scout Object Navigation Evaluation")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Evaluation failed with error:")
        print(str(e))
        print("="*60)
        import traceback
        traceback.print_exc()