#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:0
#SBATCH --job-name="CS 601.471/671 homework6"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=16G

source ~/.bashrc
module load anaconda
conda activate ssm_hw6

# python generate.py --prompt "What is the Capital of Argentina?" --checkpoint_path checkpoints/Qwen/Qwen2.5-7B-Instruct/model.pth --num_samples 1 --max_new_tokens 50 --device cuda
python generate.py --prompt "What is the Capital of Argentina?" --checkpoint_path checkpoints/Qwen2-0.5B-Instruct/model.pth --num_samples 1 --max_new_tokens 50 --device cuda