#!/bin/sh
#BSUB -q gpuv100
#BSUB -J BatchJob24sep25
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "select[gpu32gb] rusage[mem=32768] span[hosts=1]"
#BSUB -u kaaso@space.dtu.dk
#BSUB -B 
#BSUB -N 
#BSUB -o data/batch_runs/gpu_%J.out
#BSUB -e data/batch_runs/gpu_%J.err


module load cuda/12.4
source /work3/kaaso/miniconda3/etc/profile.d/conda.sh
conda activate openSARship2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 train_LSSD.py --BATCH_SIZE 35 