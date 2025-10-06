import os
import sys
import json
import yaml
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
        habitat_config_path = (
            "benchmark/nav/objectnav/objectnav_hm3d_2022_rgbd_with_semantic.yaml"  # V1
        )
    elif project_config.DATASET == "habitat_objnav_2023":
        habitat_config_path = (
            "benchmark/nav/objectnav/objectnav_hm3d_rgbd_with_semantic.yaml"  # V2
        )
    elif project_config.DATASET == "goat":
        habitat_config_path = "benchmark/nav/goat/goat_hm3d_rgbd_with_semantic.yaml"

    habitat_config = get_config(habitat_config_path)
    config = DictConfig({**habitat_config, **project_config})
    config.PRINT_IMAGES = 1
    config.habitat.simulator.agents.agent0 = config.habitat.simulator.agents.pop(
        "main_agent"
    )
    config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.min_depth = 0.0
    config.habitat.simulator.agents_order = ["agent0"]
    if project_config.DATASET == "goat":
        config.habitat.dataset.split = "val_seen"
    else:
        config.habitat.dataset.split = "val"

    with open("./merged_config.yaml", "w") as f:
        f.write(yaml.dump(OmegaConf.to_container(config), sort_keys=False))

    all_scenes = os.listdir(
        os.path.dirname(
            config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
        )
        + "/content/"
    )
    all_scenes = sorted([x.split(".")[0] for x in all_scenes if x.endswith(".json.gz")])
    logger.debug(f"All scenes: {all_scenes}")

    config.habitat.dataset.content_scenes = all_scenes[:2]
    # downward_steps = ["7MXmsvcQjpJ", "6s7QHgap2fW", "BAbdmeyTvMZ"]

    return config


def save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs):
    obs_tasks = []
    for task in obs.task_observations["tasks"]:
        obs_task = {}
        for key, value in task.items():
            if key == "image":
                continue
            obs_task[key] = value
        obs_tasks.append(obs_task)

    ep_results = {}
    ep_results["metrics"] = all_subtask_metrics
    ep_results["total_num_steps"] = ep_step
    ep_results["sub_task_timesteps"] = agent.sub_task_timesteps
    ep_results["tasks"] = obs_tasks

    for metric in all_subtask_metrics[0].keys():
        values = [y[metric] for y in all_subtask_metrics]
        if isinstance(values[0], dict):
            for i, v in enumerate(values):
                values[i] = min(v.values())


        ep_results[f"{metric}_mean"] = np.round(
            np.nanmean(values),
            4,
        )
        ep_results[f"{metric}_median"] = np.round(
            np.nanmedian(values),
            4,
        )

    results[f"{env.scene_id}_{env.episode_id}"] = ep_results
    with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
        json.dump(results, fp, indent=4)

    stats = {}

    for metric in all_subtask_metrics[0].keys():
        values = [
            y[metric]
            for scene_ep_id in results.keys()
            for y in results[scene_ep_id]["metrics"]
        ]
        if isinstance(values[0], dict):
            for i, v in enumerate(values):
                values[i] = min(values[i].values())
        stats[f"{metric}_mean"] = np.round(
            np.nanmean(values),
            4,
        )
        stats[f"{metric}_median"] = np.round(
            np.nanmedian(values),
            4,
        )

    with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
        json.dump(stats, fp, indent=4)


if __name__ == "__main__":
    args = read_args()

    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    logger = get_logger()

    config =  read_configs(args)

    logger.info("Starting code")
    logger.info(f"Using scenes: {config.habitat.dataset.content_scenes}")

    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)
    agent = GoatAgent(
        config, env.semantic_category_mapping
    )

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    results = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        if os.path.exists(os.path.join(results_dir, "per_episode_metrics.json")):
            with open(os.path.join(results_dir, "per_episode_metrics.json"), "r") as fp:
                results = json.load(fp)
        logger.info(f"Evaluating scene {env.scene_id} episode {env.episode_id}")
        if f"{env.scene_id}_{env.episode_id}" in list(results.keys()):
            continue
        env.reset_visualization()
        agent.reset(env.scene_id, env.episode_id, env.current_task_idx)

        ep_step = 0
        all_subtask_metrics = []
        pbar = tqdm(
            total=config.AGENT.max_steps, file=sys.__stdout__, dynamic_ncols=True
        )

        old_task_idx = -1
        while not env.episode_over:
            if env.current_task_idx != old_task_idx:
                logger.info(
                    f"Starting task {env.current_task_idx} in scene {env.scene_id} episode {env.episode_id}"
                )
                old_task_idx = env.current_task_idx
                pbar.set_description(
                    f"{env.scene_id}_{env.episode_id}_{env.current_task_idx}"
                )
            ep_step += 1
            logger.info(
                f"-------------------- Episode step {ep_step} --------------------"
            )
            logger.debug(f"Agent state: {env.habitat_env.sim.agents[0].get_state()}")
            env.timestep = agent.get_subtask_timestep() + 1
            obs = env.get_observation()

            agent.update_state(obs)
            action, info, stuck = agent.act()
            if stuck: 
                action = DiscreteNavigationAction.STOP

            logger.info(f"Action taken: {action}")
            env.apply_action(action, info)
            pbar.update(1)

            if action == DiscreteNavigationAction.STOP:
                ep_metrics = env.get_episode_metrics()
                agent.reset_sub_episode()
                ep_metrics.pop("goat_top_down_map", None)
                logger.info("-------------------------")
                logger.info(
                    f"{env.scene_id}_{env.episode_id}_{env.current_task_idx} {ep_metrics}"
                )
                logger.info("-------------------------")

                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent._reset_vis_dir(
                        env.scene_id, env.episode_id, env.current_task_idx
                    )
                    env.visualizer.set_vis_dir(
                        env.scene_id,
                        f"{env.episode_id}_{env.current_task_idx}",
                    )
                    pbar.reset()

        logger.info(
            f"------------------------ Episode {env.scene_id} {env.episode} over ------------------------"
        )
        pbar.close()
        save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs)
