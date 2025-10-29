import os
import sys
import json
import yaml
from typing import List
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
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import MultiAgentHabitatGoatEnv

from eval_episode import read_args, save_results


def read_configs(args):
    project_config = OmegaConf.load(args.project_config_path)
    if project_config.DATASET == "habitat_objnav_2023":
        habitat_config_path = "benchmark/nav/objectnav/multiagent_objectnav_hm3d_rgbd_with_semantic.yaml" # V2
    else:
        raise NotImplementedError("Support for other datasets is not tested.")

    

    habitat_config = get_config(habitat_config_path)
    config = DictConfig({**habitat_config, **project_config})


    config.PRINT_IMAGES = 1
    config.habitat.dataset.split = "val"
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0


    base_agent_conf = config.habitat.simulator.agents.pop("main_agent")
    import copy
    agents = []
    for i in range(config.NUM_AGENTS):
        agents.append(f"agent{i}")
        agent_conf = copy.deepcopy(base_agent_conf)

        config.habitat.simulator.agents[f"agent{i}"] = agent_conf
    config.habitat.simulator.agents_order = agents
    
    with open("./ma_merged_config.yaml", "w") as f:
        f.write(yaml.dump(OmegaConf.to_container(config), sort_keys=False))


    all_scenes = os.listdir(
        os.path.dirname(
            config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
        )
        + "/content/"
    )
    all_scenes = sorted([x.split(".")[0] for x in all_scenes if x.endswith(".json.gz")])
    logger.debug(f"All scenes: {all_scenes}")

    config.habitat.dataset.content_scenes = all_scenes[:5]
    # downward_steps = ["7MXmsvcQjpJ", "6s7QHgap2fW", "BAbdmeyTvMZ"]

    return config
    


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
    env = MultiAgentHabitatGoatEnv(habitat_env, config=config)
    agents: List[GoatAgent] = []
    for i in range(config.NUM_AGENTS):
        agents.append(GoatAgent(config, env.semantic_category_mapping, i))

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    results = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        logger.info(f"Evaluating scene {env.scene_id} episode {env.episode_id}")
        if os.path.exists(os.path.join(results_dir, "per_episode_metrics.json")):
            with open(os.path.join(results_dir, "per_episode_metrics.json"), "r") as fp:
                results = json.load(fp)
        if f"{env.scene_id}_{env.episode_id}" in list(results.keys()):
            continue
        env.reset_visualization()
        for agent in agents:
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
            observations = env.get_observation()

            actions = []
            infos = []
            stucks = []
            for agent, obs in zip(agents, observations):
                agent.update_state(obs)

            for agent in agents:
                other_agents = list(filter(lambda x: x.agent_id != agent.agent_id, agents))
                action, info, stuck = agent.act(other_agents)
                
                actions.append(action)
                infos.append(info)
                stucks.append(stuck)
            
            if all(stucks):
                actions = [DiscreteNavigationAction.STOP]*2

            logger.info(f"Actions taken: {actions}")
            env.apply_action(actions, info=infos)
            pbar.update(1)

            if DiscreteNavigationAction.STOP in actions:
                for agent in agents:
                    agent.reset_sub_episode()
                ep_metrics = env.get_episode_metrics()
                ep_metrics.pop("goat_top_down_map", None)
                logger.info("-------------------------")
                logger.info(
                    f"{env.scene_id}_{env.episode_id}_{env.current_task_idx} {ep_metrics}"
                )
                logger.info("-------------------------")

                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    for agent in agents:
                        agent.reset_vis_dir(
                            env.scene_id, env.episode_id, env.current_task_idx
                        )
                    env.visualizer.set_vis_dir(
                        env.scene_id,
                        f"{env.episode_id}_{env.current_task_idx}",
                    )
                    pbar.reset()

        logger.info(
            f"------------------------ Episode {env.scene_id} {env.episode.episode_id} over ------------------------"
        )
        pbar.close()
        results = save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs)
