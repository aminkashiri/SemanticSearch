import yaml
from omegaconf import OmegaConf
from habitat.config.default import get_config, read_write

from habitat_baselines.config.default import _BASELINES_CFG_DIR
cfg = get_config("/home-robot/src/third_party/habitat-lab/habitat-baselines/habitat_baselines/config/goat/modular_goat_hm3d_fixed.yaml", configs_dir=_BASELINES_CFG_DIR)  
with open("./merged_config.yaml", "w") as f:
    f.write(yaml.dump(OmegaConf.to_container(cfg), sort_keys=False))