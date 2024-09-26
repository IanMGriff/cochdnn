#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_%j.out
#SBATCH --error=outLogs/eval_jsin_%j.err
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus=1

#SBATCH --mem=16Gb
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

module load cuda cudnn nccl

conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# python3 lightning_scripts/eval_jsin.py --config lightning_scripts/configs/word_audioset_resnet50.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192


# python3 lightning_scripts/eval_jsin.py --config lightning_scripts/configs/word_audioset_resnet50_lower_lr.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192

python3 lightning_scripts/eval_jsin.py --config lightning_scripts/configs/word_audioset_resnet50_lower_lr_slower_schedule_lower_task_weight.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192

