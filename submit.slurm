#!/bin/bash -l
#SBATCH --job-name=DiffusionAutoEncoder24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --nodes=11
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu_32gb'
#SBATCH --output=/work/cseos2g/papapalpi/DeepDriveStuff/bdd100k/images/logs/%x-%j.out


# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# (optional) set the network interface name manually
# export NCCL_SOCKET_IFNAME=lo

# load your Python environment, for example with conda
conda activate your_env_name

# (optional) run script with fake data for debugging
# srun python train.py --trainer.num_nodes 2 --trainer.gpus 8

# run full training with ImageNet data already downloaded
#srun python train.py --trainer.num_nodes 2 --trainer.gpus 8 \
#--data_path path/to/dataset

export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.8/site-packages/:$PYTHONPATH"
srun singularity exec --nv $WORK/torchgan_container_old_new_latest.sif python3 train.py fit --trainer.num_nodes 11 --trainer.gpus 2
