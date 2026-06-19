#!/bin/bash
#SBATCH --job-name=cirh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="yesminaelarkoubi@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge

module load ALICE/default
module load slurm
# module load CUDA/12.3.2
# module load cuDNN/8.9.7.29
# module load GCCcore/12.2.0
module load CUDA/12.3.2
module load GCC/11.3.0
module load Python/3.10.8-GCCcore-12.2.0

# Source the Python virtual environment
source $HOME/cirh_dir/CIRH/test_venv/bin/activate


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES
nvidia-smi

# Run the file
echo "[$SHELL] ## Run script"
python3 main_mir.py
echo "[$SHELL] ## Script finished"
echo "[$SHELL] #### Finished Python test. Have a nice day"