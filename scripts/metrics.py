import json
import os
import numpy as np

def read_metrics(folder):
    file_path = os.path.join("outputs", folder, "results", "stretch_openvocab__0", "per_episode_metrics.json")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

metrics_default = read_metrics("default")
metrics_fix = read_metrics("fixed")

default_success_rate = []
fix_success_rate = []

default_SPL = []
fix_SPL = []

for episode in metrics_default.keys():
    ep_metrics = metrics_default[episode]["metrics"]
    ep_metrics_fix = metrics_fix[episode]["metrics"]
    ep_success_rate = [x["goat_sub-task_success"] for x in ep_metrics]
    ep_success_rate_fix = [x["goat_sub-task_success"] for x in ep_metrics_fix]

    ep_SPL = [x["goat_sub-task_spl"] for x in ep_metrics]
    ep_SP_fix = [x["goat_sub-task_spl"] for x in ep_metrics_fix]

    default_success_rate.extend(ep_success_rate)
    fix_success_rate.extend(ep_success_rate_fix)
    default_SPL.extend(ep_SPL)
    fix_SPL.extend(ep_SP_fix)

    print(f"\n------- Episode {episode}:")
    print(f"  Default Success Rate: {np.mean(ep_success_rate)}")
    print(f"  Fix Success Rate: {np.mean(ep_success_rate_fix)}")
    print(f"  Default SPL: {np.mean(ep_SPL)}")
    print(f"  Fix SPL: {np.mean(ep_SP_fix)}")

print(f"Total")
print(f"  Default Success Rate: {np.mean(default_success_rate)}")
print(f"  Fix Success Rate: {np.mean(fix_success_rate)}")
print("")
print(f"  Default SPL: {np.mean(default_SPL)}")
print(f"  Fix SPL: {np.mean(fix_SPL)}")