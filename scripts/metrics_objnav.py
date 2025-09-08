import json
import os
import numpy as np

def read_metrics(folder):
    # file_path = os.path.join("outputs", folder, "results", "stretch_openvocab__0", "per_episode_metrics.json")
    file_path = os.path.join("datadump", "results", folder, "per_episode_metrics.json")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

metrics = read_metrics("objnav_hm3d_2023_closest_viewpoint_0")
# metrics = read_metrics("objnav_hm3d_2022_closest_viewpoint_0")
# metrics = read_metrics("objectnav_hm3d_challenge2022_val_0")

results = {episode.split("_")[0]: {"spl":[], "success":[]} for episode in metrics.keys()}

for episode in metrics.keys():
    scene = episode.split("_")[0]
    ep_metrics = metrics[episode]["metrics"][0]

    results[scene]["spl"].append(ep_metrics["spl"])
    results[scene]["success"].append(ep_metrics["success"])

for scene in results.keys():
    spl = np.mean(results[scene]["spl"])
    success = np.mean(results[scene]["success"])
    print(f"\n------- Scene {scene}:")
    print(f"  Success Rate: {success}")
    print(f"  SPL: {spl}")
