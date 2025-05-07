FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

# Install git-lfs
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs


# Set git config and pull goat-sim branch from your GitHub repo
WORKDIR /home-robot
RUN git fetch && \
    git config pull.rebase false && \
    git checkout goat-sim && \
    git remote add myorigin https://github.com/aminkashiri/SemanticSearch.git && \
    git pull myorigin goat-sim

# Use bash shell for conda activation
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=home-robot

# Activate conda and install everything
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate home-robot && \
    git submodule update --init --recursive \
        src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet \
        src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/ \
        src/third_party/habitat-lab/ && \
    export TORCH_CUDA_ARCH_LIST="8.6" && \
    pip install -e src/third_party/detectron2 && \
    pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt && \
    pip install -e src/third_party/habitat-lab/habitat-lab && \
    pip install -e src/third_party/habitat-lab/habitat-baselines && \
    pip install "gym>=0.25" && \
    pip install bresenham && \
    pip install gdown && \
    gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe

# Download pretrained Detic model
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate

# Download goat benchmark dataset and unzip
RUN unzip goat-bench.zip && \
    mv data/datasets/goat_bench data/datasets/goat_openvocab && \
    mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed

# Download HM3D dataset
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate home-robot && \
    yes | python -m habitat_sim.utils.datasets_download \
    --username 2523e65432178e8d \
    --password c5cd9238658cf8c9c81676965a49b859 \
    --uids hm3d_full