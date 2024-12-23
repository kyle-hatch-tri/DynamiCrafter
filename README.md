# Train Video-Generative High Level Policies for GHIL-Glue

This fork of the [DynamicCrafter](https://github.com/Doubiiu/DynamiCrafter) repo contains code for training video-generative high level policies for [GHIL-Glue](https://github.com/kyle-hatch-tri/ghil-glue.git).



## Installation
```
conda create -n dynamicrafter python=3.8.5
conda activate dynamicrafter
pip install -r requirements.txt
```
For troubleshooting see https://github.com/Doubiiu/DynamiCrafter


## Download checkpoints

The trained diffusion model checkpoints can be downloaded from https://huggingface.co/kyle-hatch-tri/ghil-glue-checkpoints


## Data

1. Raw datasets can be downloaded following the instructions from the [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) and the [CALVIN](https://github.com/mees/calvin) repos. 
2. These should then be processed using the instructions from our fork of the [BridgeData V2](repo)
3. The processed datasets can then be converted to the `webvid_format` using `preprocess_bridge_data.py` and `preprocess_calvin_data.py`, which is the format used by this repo for training.

## Training

Local training on the CALVIN dataset can be launched with the following command: `bash configs/training_256_v1.0/run.sh 0`. See the `configs` folder  for other training configurations.
Note that to reproduce the training results in the paper, the diffusion model weights should be initalized from the original checkpoints listed in the [DynamicCrafter](https://github.com/Doubiiu/DynamiCrafter) repo.