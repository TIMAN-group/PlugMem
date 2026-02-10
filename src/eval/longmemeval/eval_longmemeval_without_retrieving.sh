#!/bin/bash
#SBATCH --account=bcjf-delta-cpu   # 如果没有 GPU quota，也可以改成你的 CPU 项目账号
#SBATCH --partition=cpu             # CPU partition
#SBATCH --nodes=1                   # 1 个节点
#SBATCH --ntasks=1                  # 1 个任务
#SBATCH --cpus-per-task=4           # CPU 核心数，可以改
#SBATCH --mem=32G                    # 内存大小
#SBATCH --time=20:00:00             # 最大运行时间

python eval_longmemeval_without_retrieving.py