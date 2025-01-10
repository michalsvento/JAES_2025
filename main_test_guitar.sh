#!/bin/bash

# ---------- Slurm and env setup ----------

# time setup
##SBATCH  --time=0-23:29:59
#SBATCH  --time=0-03:59:59
##SBATCH  --time=01:59:59

# CPU, GPU, memory setup
#SBATCH --mem=30G

#SBATCH --cpus-per-task=1
#SBATCH  --gres=gpu:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --exclude=gpu11

#SBATCH --array=[1]
#SBATCH --job-name=Career_clipfree
#SBATCH --output=/scratch/work/%u/projects/diffusion_distortion/experiments/train_%j.out

export HYDRA_FULL_ERROR=1 

# Set your conda environment
module load mamba
#source activate /scratch/work/molinee2/conda_envs/env2024
source activate  /scratch/work/sventom2/.conda_envs/myenv


# Main config file
conf=conf_guitar.yaml

# Select one config file for specific task
#########################################
#-------- Guitar + hard clipping -------#
#########################################

tester=hardclip_test
target_SDR=1

#########################################
#-------- Guitar + soft clipping -------#
#########################################

#tester=softclip_test
target_SDR=1

####################################
#---------- Guitar + HWR ----------#
####################################

#tester=hwr_test

#########################################
#---------- Guitar + foldback ----------#
#########################################

#tester=foldback_test

#############################################
#---------- Guitar + quantization ----------#
#############################################

#tester=quantization_test

##########################################
#---------- Guitar + carbonmic ----------#
##########################################

#tester=carbon_mic_test

# Method selection
NLD="CCR" #"MLP" #"sumtanh"

# Dataset and network config
dset=Career_B_test
network=cqtdiff+_44k_32binsoct

#guitar checkpoint
# Here add your path to the checkpoint
ckpt="checkpoints/guitar_IDMT_Career_44k/guitar_CareerSG_44k_6s-325000.pt"

# Name for the experiment
n="experiment"

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
