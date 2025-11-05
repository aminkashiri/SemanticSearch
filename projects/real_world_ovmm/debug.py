#!/usr/bin/env python
"""
Simple test to check if Detic is detecting objects and map is updating.
No config modifications needed.
"""

import sys
sys.path.append('/home/atharva/SemanticSearch/src/home_robot')

import numpy as np
import rospy
from home_robot_hw.env.scout_object_nav_env import ScoutObjectNavEnv
from home_robot_hw.utils.config import load_config

def main():
    print("="*60)
    print("SIMPLE NAVIGATION TEST")
    print("="*60)
    
    rospy.init_node("simple_test")
    
    # Load config
    config = load_config()
    
    # Create environment
    print("\n1. Creating environment...")
    env = ScoutObjectNavEnv(
        config=config,
        cat_map_file="projects/real_world_ovmm/configs/example_cat_map.json"
    )
    
    print(f"   ✅ Environment created")
    print(f"   Vocabulary size: {len(env.goal_options)}")
    
    # Set a goal
    print("\n2. Setting goal to 'cup'...")
    if env.set_goal("cup"):
        print(f"   ✅ Goal set: {env.current_goal_name} (ID: {env.current_goal_id})")
    else:
        print(f"   ❌ Could not set goal 'cup'")
        print(f"   Available: {env.goal_options[:20]}")
        return
    
    # Get one observation
    print("\n3. Getting observation...")
    obs = env.get_observation()
    
    print(f"   ✅ Observation received")
    print(f"   RGB shape: {obs.rgb.shape}")
    print(f"   Depth shape: {obs.depth.shape}")
    print(f"   Depth range: {obs.depth.min():.3f}m to {obs.depth.max():.3f}m")
    
    # Check semantic detection
    print("\n4. Checking Detic detection...")
    unique_labels = np.unique(obs.semantic)
    print(f"   Detected semantic IDs: {unique_labels}")
    
    # Print vocabulary mapping for detected IDs
    print(f"\n   Vocabulary mapping:")
    for i, cat in enumerate(env.goal_options[:50]):  # Show first 50
        marker = " 🎯" if i == env.current_goal_id else ""
        print(f"      Index {i:3d} = {cat}{marker}")
    
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        print("   ⚠️  WARNING: Only 'other' (ID=0) detected!")
        print("   Possible issues:")
        print("      - Detic confidence threshold too high")
        print("      - Camera not seeing any objects")
        print("      - Vocabulary mismatch")
    else:
        print(f"\n   ✅ Detecting {len(unique_labels)} different categories")
        for label_id in unique_labels:
            pixel_count = (obs.semantic == label_id).sum()
            pct = 100 * pixel_count / obs.semantic.size
            if label_id < len(env.goal_options):
                label_name = env.goal_options[label_id]
                is_goal = " 🎯 GOAL!" if label_id == env.current_goal_id else ""
                print(f"      ID {label_id:3d} ({label_name:20s}): {pixel_count:6d} pixels ({pct:5.2f}%){is_goal}")
    
    # Check if goal is visible
    print("\n5. Checking if goal is visible...")
    goal_visible = env.current_goal_id in unique_labels
    if goal_visible:
        goal_pixels = (obs.semantic == env.current_goal_id).sum()
        print(f"   ✅ GOAL IS VISIBLE!")
        print(f"   Goal '{env.current_goal_name}' detected: {goal_pixels} pixels")
    else:
        print(f"   ❌ Goal '{env.current_goal_name}' (ID {env.current_goal_id}) NOT visible")
        print(f"   Robot needs to explore to find it")
    
    # Check task observations
    print("\n6. Checking task observations...")
    print(f"   goal_name: {obs.task_observations['goal_name']}")
    print(f"   goal_id: {obs.task_observations['goal_id']}")
    print(f"   tasks: {obs.task_observations['tasks']}")
    
    # Check GPS/Compass
    print("\n7. Checking pose...")
    print(f"   GPS: {obs.gps}")
    print(f"   Compass: {obs.compass}")
    
    # Test movement
    print("\n8. Testing basic movement...")
    from home_robot.core.interfaces import DiscreteNavigationAction
    
    print("   Moving forward 25cm...")
    env.apply_action(DiscreteNavigationAction.MOVE_FORWARD)
    
    # Get another observation
    obs2 = env.get_observation()
    
    # Check if pose changed
    pose_changed = not np.allclose(obs.gps, obs2.gps) or not np.allclose(obs.compass, obs2.compass)
    if pose_changed:
        print(f"   ✅ Robot moved!")
        print(f"   New GPS: {obs2.gps}")
        print(f"   Delta: {obs2.gps - obs.gps}")
    else:
        print(f"   ⚠️  Robot did not move (or odometry not updating)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Summary
    print("\n📊 SUMMARY:")
    issues = []
    
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        issues.append("❌ Detic not detecting objects")
    else:
        print("✅ Detic is detecting objects")
    
    if goal_visible:
        print("✅ Goal is visible in current view")
    else:
        print("⚠️  Goal not visible (normal - need to explore)")
    
    if pose_changed:
        print("✅ Robot can move and odometry works")
    else:
        issues.append("❌ Robot movement or odometry not working")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print("\nNext steps:")
        print("   1. If Detic not detecting: Lower confidence_threshold in eval.yaml")
        print("   2. If robot not moving: Check /cmd_vel topic and navigation")
        print("   3. Try running full eval with --max-num-steps 10 to see behavior")
    else:
        print("\n✅ ALL CHECKS PASSED - System appears to be working!")
        print("   You can now run full navigation test")

if __name__ == "__main__":
    main()