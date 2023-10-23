#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  
#SBATCH --mem-per-cpu=48G 
#SBATCH --time=4-0:0:0
#SBATCH --partition=e7 
#SBATCH --job-name=yolino_tusimple
#SBATCH --output=multinode_log_sbatch_%n.log
#SBATCH --wait-all-nodes=1
#SBATCH --reservation=e7

module load CUDA/11.3.1

nodelist=($(scontrol show hostname $SLURM_JOB_NODELIST))
current_dist_url=$(cat /etc/hosts | grep $(hostname) | awk '{print $2}' | head -n 1)
first_node_dist_url=$(cat /etc/hosts | grep ${nodelist[0]} | awk '{print $2}' | head -n 1)

const_args="CUDA_VISIBLE_DEVICES="0" python ../yolino/src/yolino/train.py --gpu "
srun --nodes=1 singularity exec --nv yolino_container.sif ${const_args} &
wait