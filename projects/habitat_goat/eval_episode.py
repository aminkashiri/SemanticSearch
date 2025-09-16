import os
import sys
import json
import pprint
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from habitat.core.env import Env
from omegaconf import DictConfig, OmegaConf
from habitat.config.default import get_config
from home_robot.utils.logger import get_logger
from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv

def read_args():
    parser = argparse.ArgumentParser()
    project_config_default = "projects/habitat_goat/configs/agent/hm3d_eval_new.yaml"
    parser.add_argument(
        "--project_config_path",
        type=str,
        default=project_config_default,
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args

def read_configs(args):
    project_config = OmegaConf.load(args.project_config_path)
    if project_config.DATASET == "habitat_objnav_2022":
        habitat_config_path = "benchmark/nav/objectnav/objectnav_hm3d_2022_rgbd_with_semantic.yaml" # V1
        stuck_metric = "distance_to_goal"
    elif project_config.DATASET == "habitat_objnav_2023":
        habitat_config_path = "benchmark/nav/objectnav/objectnav_hm3d_rgbd_with_semantic.yaml" # V2
        stuck_metric = "distance_to_goal"
    elif project_config.DATASET == "goat":
        habitat_config_path = "benchmark/nav/goat/goat_hm3d_rgbd_with_semantic.yaml"
        stuck_metric = "goat_distance_to_sub-goal"

    habitat_config = get_config(habitat_config_path)
    config = DictConfig({**habitat_config, **project_config})
    config.PRINT_IMAGES = 1
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0
    if project_config.DATASET == "goat":
        config.habitat.dataset.split = "val_seen"

    all_scenes = os.listdir(
        os.path.dirname(
            config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
        )
        + "/content/"
    )
    all_scenes = sorted([x.split(".")[0] for x in all_scenes if x.endswith(".json.gz")])
    logger.debug(f"All scenes: {all_scenes}")

    config.habitat.dataset.content_scenes = all_scenes[:]
    # downward_steps = ["7MXmsvcQjpJ", "6s7QHgap2fW", "BAbdmeyTvMZ"]

    return config, stuck_metric
    


if __name__ == "__main__":
    args = read_args()

    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    logger = get_logger()

    config, stuck_metric = read_configs(args)

    logger.info("Starting code")
    logger.info(f"Using scenes: {config.habitat.dataset.content_scenes}")

    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)
    agent = GoatAgent(config=config, semantic_category_mapping=env.semantic_category_mapping)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    task_type = config.habitat.task.type

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        agent.reset()
        stop = False

        old_distance_to_goal = None
        ctr = 0

        ep_step = 0

        scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
        episode = env.habitat_env.current_episode
        episode_id = episode.episode_id

        logger.info(f"Evaluating scene {scene_id} episode {episode_id}")

        if os.path.exists(os.path.join(results_dir, "per_episode_metrics.json")):
            with open(os.path.join(results_dir, "per_episode_metrics.json"), "r") as fp:
                metrics = json.load(fp)

        scene_ep_pairs = list(metrics.keys())
        if f"{scene_id}_{episode_id}" in scene_ep_pairs:
            continue

        current_task_idx = env.habitat_env.task.current_task_idx if task_type == "Goat-v1" else 0 
        agent.planner.set_vis_dir(
            scene_id, f"{episode_id}_{current_task_idx}"
        )
        agent.imagenav_visualizer.set_vis_dir(
            f"{scene_id}_{episode_id}_{current_task_idx}"
        )
        agent.matching.set_vis_dir(
            f"{scene_id}_{episode_id}_{current_task_idx}"
        )
        env.visualizer.set_vis_dir(
            scene_id, f"{episode_id}_{current_task_idx}"
        )

        all_subtask_metrics = []
        pbar = tqdm(total=config.AGENT.max_steps, file=sys.__stdout__, dynamic_ncols=True)

        old_task_idx = -1
        while not env.episode_over:
            current_task_idx = env.habitat_env.task.current_task_idx if task_type == "Goat-v1" else 0 
            if current_task_idx != old_task_idx:
                logger.info(
                    f"Starting task {current_task_idx} in scene {scene_id} episode {episode_id}"
                )
                old_task_idx = current_task_idx
            ep_step += 1
            logger.info(f"-------------------- Episode step {ep_step} --------------------")
            logger.debug(f"Agent state: {env.habitat_env.sim.agents[0].get_state()}")
            env.timestep = agent.get_subtask_timestep() + 1
            obs = env.get_observation()
            if ep_step == 1:
                obs_tasks = []
                for task in obs.task_observations["tasks"]:
                    obs_task = {}
                    for key, value in task.items():
                        if key == "image":
                            continue
                        obs_task[key] = value
                    obs_tasks.append(obs_task)

                logger.debug(f"tasks are: {pprint.pformat(obs_tasks)}")

            action, info = agent.act(obs, stop)
            logger.info(f"Action taken: {action}")
            env.apply_action(action, info=info)
            pbar.set_description(f"{scene_id}_{episode_id}_{current_task_idx}")
            pbar.update(1)

            if (
                env.get_episode_metrics()[stuck_metric]
                == old_distance_to_goal
            ):
                ctr += 1

                if ctr > 20:
                    logger.info("Agent was stuck. Stopping episode.")
                    # action = DiscreteNavigationAction.STOP
                    stop = True
                    ctr = 0
            else:
                ctr = 0

            old_distance_to_goal = env.get_episode_metrics()[stuck_metric]

            if action == DiscreteNavigationAction.STOP:
                stop = False
                ep_metrics = env.get_episode_metrics()
                ep_metrics.pop("goat_top_down_map", None)
                logger.info("-------------------------")
                logger.info(f"{scene_id}_{episode_id}_{current_task_idx} {ep_metrics}")
                logger.info("-------------------------")

                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent.imagenav_visualizer.set_vis_dir(
                        f"{scene_id}_{episode_id}_{current_task_idx}"
                    )
                    agent.matching.set_vis_dir(
                        f"{scene_id}_{episode_id}_{current_task_idx}"
                    )
                    agent.planner.set_vis_dir(
                        scene_id,
                        f"{episode_id}_{current_task_idx}",
                    )
                    env.visualizer.set_vis_dir(
                        scene_id,
                        f"{episode_id}_{current_task_idx}",
                    )
                    pbar.reset()

        pbar.close()

        ep_metrics = env.get_episode_metrics()
        scene_ep_id = f"{scene_id}_{episode_id}"
        metrics[scene_ep_id] = {"metrics": all_subtask_metrics}
        metrics[scene_ep_id]["total_num_steps"] = ep_step
        metrics[scene_ep_id]["sub_task_timesteps"] = agent.sub_task_timesteps[0]
        metrics[scene_ep_id]["tasks"] = obs_tasks

        try:
            for metric in list(metrics.values())[0]["metrics"][0].keys():
                metrics[scene_ep_id][f"{metric}_mean"] = np.round(
                    np.nanmean(
                        np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                    ),
                    4,
                )
                metrics[scene_ep_id][f"{metric}_median"] = np.round(
                    np.nanmedian(
                        np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                    ),
                    4,
                )
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()

        logger.info(f"------------------------ Episode {scene_ep_id} over ------------------------")

        with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        stats = {}

        for metric in list(metrics.values())[0]["metrics"][0].keys():
            stats[f"{metric}_mean"] = np.round(
                np.nanmean(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )
            stats[f"{metric}_median"] = np.round(
                np.nanmedian(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )

        with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
