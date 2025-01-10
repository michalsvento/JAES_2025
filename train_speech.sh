#!/bin/bash

module load gcc

export HYDRA_FULL_ERROR=1 

module load mamba
source activate /scratch/work/molinee2/conda_envs/envpython311


conf=conf_VCTK_16k.yaml
dset=vctk_16k_4s

exp=speech_16k_4s

network=ncsnpp

n="train_speech"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT

python train.py --config-name=$conf\
  model_dir=$PATH_EXPERIMENT \
  dset=$dset \
  network=$network \
  exp=$exp

