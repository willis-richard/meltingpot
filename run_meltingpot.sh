#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=meltingpot
#SBATCH --time=2-00
#SBATCH --mem=120G
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 16
#SBATCH --nodes=1
# works
#SBATCH --nodelist=erc-hpc-comp031,erc-hpc-comp033,erc-hpc-comp034,erc-hpc-comp038,erc-hpc-comp039
# testing
# #SBATCH --nodelist=erc-hpc-comp033,erc-hpc-comp037

# #SBATCH --tasks-per-node=1
# #SBATCH --ntasks 1
# #SBATCH --cpus-per-task=64
# #SBATCH --mem-per-cpu=8G
# #SBATCH --gpus-per-task=0
# #SBATCH --gres=gpu
# #SBATCH --exclusive



echo "Running on node: $(hostname)"

source ~/mambaforge/etc/profile.d/conda.sh
conda activate paper_2

# module load cuda/11.8.0-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
# module load cudnn/8.7.0.84-11.8-gcc-13.2.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDNN_HOME/lib64

export PYTHONPATH="/users/k21075402/repo/meltingpot/"
cd $PYTHONPATH

# commons harvest
# 39596646  3f82a7f6  b484e613  d1a59a0e  ebcd23ed
# 41697eb3 d1ec3174 eb100aee b0569bd9 68238f61
# 30ca9764
# shrooms
# ec5de638 - delete
# 8de7e630 b7c04d31 ca8d5c9d cb2f86c9 fba0fc0e
# 51727c4c scratch

python3 reward_transfer/experiments.py \
  --trial_id fba0fc0e \
  --substrate externality_mushrooms__dense \
  --n_iterations 300 \
  --framework tf2 \
  --num_cpus 16 \
  --num_gpus 1 \
  --local_dir $SCRATCH/tf_2_14 \
  --rollout_workers 15 \
  --episodes_per_worker 2 \
  --max_concurrent_trials 1 \
  --wandb mushroom-2-14 \
  --training training \
  --n 5 \
  --training_mode independent \
  --tmp_dir $SCRATCH/tmp # 1>/dev/null 2>&1


# CHECKPOINT_PATH=$SCRATCH/optuna/single/commons_harvest__open/PPO_ec9c9364_8000,9,0.00033,1,0.9,0.35,2_89_AlgorithmConfig__prior_exploration_config=None,disable_action_flattening=False,disable_2024-05-10_06-15-55/checkpoint_000000/policies/default
# CHECKPOINT_PATH=$SCRATCH/optuna/random/commons_harvest__open/PPO_24778620_14000,15,0.0009,0.95,0.7,0.4,9_11_AlgorithmConfig__prior_exploration_config=None,disable_action_flattening=False,disa_2024-05-10_21-43-19/checkpoint_000000/policies/default
# CHECKPOINT_PATH=$SCRATCH/optuna/single/commons_harvest__open/PPO_70fe371d_single_pre-trained_7500,10,0.00011,0.99,0.85,0.25_20_AlgorithmConfig__prior_exploration_config\=None,disable_action_fl_2024-05-31_16-05-10/checkpoint_000000/policies/default

# Best optuna futher pretraining
CHECKPOINT_PATH=$SCRATCH/optuna/single/commons_harvest__open/PPO_86bacded_single_pre-trained_10000\,12\,0.00007\,0.99\,0.8\,0.3_44_AlgorithmConfig__prior_exploration_config\=None\,disable_action_fla_2024-06-01_13-49-46/checkpoint_000000/policies/default

