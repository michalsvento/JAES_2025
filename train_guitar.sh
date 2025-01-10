#!/bin/bash

module load gcc
export HYDRA_FULL_ERROR=1 

# Set your conda environment
module load mamba
source activate /scratch/work/molinee2/conda_envs/envpython311

# main config
conf=conf_guitar.yaml

#dset=EGDB_dataset #for pretraining
dset=Career_DDD

exp=Guitar_44k_6s_augmentations

network=cqtdiff+_44k_32binsoct

tester=only_unconditional_EGDB

diff_params=edm_EGDB


n="train_guitar"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT

python train.py --config-name=$conf \
  diff_params=$diff_params \
  model_dir=$PATH_EXPERIMENT \
  dset=$dset \
  network=$network \
  tester=$tester \
  exp=$exp \
  exp.batch_size=4

