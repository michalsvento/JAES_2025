#!/bin/bash


export HYDRA_FULL_ERROR=1 

module load mamba
source activate /scratch/work/molinee2/conda_envs/env2024

# Main config file
conf=conf_VCTK_16k.yaml

################################
#------- hard clipping --------#
################################

tester=hardclip_test_vctk
target_SDR=1

################################
#------- soft clipping --------#
################################

tester=softclip_test_vctk
target_SDR=1

# Method selection
NLD="CCR" #"MLP" #"sumtanh"

# Dataset and network config
dset=vctk_test_set
network=ncsnpp

# speech checkpoint
# Here add your path to the checkpoint
ckpt="checkpoints/speech_VCTK_16k/VCTK_16k_4s-510000.pt"

# Name for the experiment
n="speech_test"

# Wandb config
entity="michalsvento"
project="diffusiondistortion"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT


python main_test.py --config-name=$conf \
  tester=$tester \
  tester.checkpoint=$ckpt \
  model_dir=$PATH_EXPERIMENT \
  network=$network \
  dset=$dset \
  +gpu=0 \
  tester.blind_distortion.op_hp.NLD=$NLD \
  tester.distortion.fix_SDR.target_SDR=$target_SDR \
  logging.wandb.entity="$entity" \
  logging.wandb.project="$project"
