#!/bin/bash

# Only needed if you don't use docker file. You can create a container with image: fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023 
# and run this script inside the container.
# docker run -it --gpus all -v ~/Workspace/exploration/hm3d_dataset/hm3d-0.2:/home-robot/data/versioned_data fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023 bash

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |  bash
apt install git-lfs
git config pull.rebase false
git remote add myorigin https://github.com/aminkashiri/SemanticSearch.git
git checkout goat-sim 
git pull myorigin goat-sim

source /opt/conda/etc/profile.d/conda.sh
conda activate home-robot

git submodule update --init --recursive src/third_party/detectron2 \
    src/home_robot/home_robot/perception/detection/detic/Detic \
    src/third_party/contact_graspnet\
    src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/ \
    src/third_party/habitat-lab/

export TORCH_CUDA_ARCH_LIST="8.6" # This nvcc version doesn't support 8.9 (GTX 4090), but this still works find, with some caveats
pip install -e src/third_party/detectron2

pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
pip install "gym>=0.25" bresenham gdown
pip install sophuspy --upgrade

mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --no-check-certificate


# Update habitat-lab to goat-support tag
# cd /home-robot/src/third_party/habitat-lab
# git checkout home-robot_goat_support
# cd /home-robot



gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe

unzip goat-bench.zip 
mv data/datasets/goat_bench data/datasets/goat_openvocab 
mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed 

# Either mount the dataset, or download it:
mkdir data/scene_datasets
ln -s /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d 

git fetch --all
git checkout goat_fixed

# yes | python -m habitat_sim.utils.datasets_download --username 2523e65432178e8d --password c5cd9238658cf8c9c81676965a49b859 --uids hm3d_full