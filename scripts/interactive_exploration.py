import habitat
from habitat.config.default import get_config, read_write
from habitat_baselines.config.default import _BASELINES_CFG_DIR
import matplotlib.pyplot as plt
import numpy as np

config_path = "/home-robot/src/third_party/habitat-lab/habitat-baselines/habitat_baselines/config/goat/modular_goat_hm3d_fixed.yaml"
# scene_id = "5cdEh9F2hJL"
scene_id = "4ok3usBNeis"
episode_id = 5

cfg = get_config(config_path, configs_dir=_BASELINES_CFG_DIR)
with read_write(cfg):
    cfg.habitat.dataset.content_scenes = [scene_id]

env = habitat.Env(config=cfg)
obs = env.reset()

while not int(env.current_episode.episode_id) == episode_id:
    obs = env.reset()

print(f"Evaluating scene {scene_id} episode {episode_id}")
env._elapsed_steps = 0

action_mapping = {
    "w": "move_forward",
    "a": "turn_left",
    "d": "turn_right",
    "q": "stop"
}
plt.ion()

fig = plt.figure(figsize=(10, 12))

try:
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("+0+0")          # Move to top-left corner
    mng.window.wm_attributes("-topmost", 1) 
    mng.window.geometry("1850x2160")        # Resize window (left half of 1920x1080 screen)
except Exception as e:
    print("Could not reposition window:", e)

def show_obs(obs):
    rgb = obs["rgb"]
    depth = obs["depth"]

    plt.clf()

    # RGB
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title("RGB Observation")
    plt.axis("off")

    # Depth
    plt.subplot(2, 1, 2)
    plt.imshow(depth.squeeze(), cmap="plasma")
    plt.title("Depth Observation")
    plt.axis("off")

    plt.tight_layout()
    plt.pause(0.001)

show_obs(obs)

while True:
    key = input("Enter action [w/a/d/q]: ").lower().strip()
    if key not in action_mapping:
        print("Invalid key. Use w/a/d/q.")
        continue
    action = action_mapping[key]
    if action == "stop":
        print("Stopping.")
        break
    obs = env.step(action)
    state = env.sim.agents[0].get_state()
    print(f"Agent state is: {state.position}")
    show_obs(obs)

env.close()
plt.ioff()  
plt.close()

