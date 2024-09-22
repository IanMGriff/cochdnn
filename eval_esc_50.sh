#!/bin/bash -l
#SBATCH --job-name=eval_esc50
#SBATCH --output=outLogs/eval_esc50_%j.out
#SBATCH --error=outLogs/eval_esc50_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1

#SBATCH --mem=50Gb
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

module load cuda cudnn nccl

conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

python3 lightning_scripts/make_esc_pl_model_plots.py --config_path lightning_scripts/configs/word_audioset_resnet50.yaml \
                                   -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
                                   --model_ckpt_dir model_checkpoints \

