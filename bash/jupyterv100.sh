#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --time=24:00:00 
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=batch 

# Load environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

module load gcc 
module load cuda

source ~/miniconda3/bin/activate
conda activate seismicenv
# get tunneling info 
export XDG_RUNTIME_DIR="" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=8889
echo $node pinned to port $port 
# print tunneling instructions 

echo -e " 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop " 

# Run Jupyter 
jupyter notebook --no-browser --port=${port} --port-retries=50 --ip=${node} 
