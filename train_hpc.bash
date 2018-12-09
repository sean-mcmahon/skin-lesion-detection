#!/bin/bash -l
#PBS -N sc_mel_def
#PBS -l ngpus=1
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l gputype=M40
#PBS -l walltime=6:00:00

module load anaconda3/5.1.0

cd ~/fastai/courses/projects
source activate fastai

USEGPU='true'
if [[ $(lsb_release -si) == *"SUSE"* ]]; then
    # On HPC (probably)

    # Old GPU ID method only works on nodes with 2x GPUs
    # GPU_ID=$(nvidia-smi | awk '{ if(NR==19) if($2>0) print 0; else print 1 }')

    # New GPU ID method works on nodes with 1 or more GPUs
    PROCESSES=$((nvidia-smi -q -d pids | grep Processes) | awk '{printf "%sHereBro ",$3}')
    ind=0
    GPU_ID=-1
    for process in $PROCESSES; do
        echo $process
        if [[ "$process" == "NoneHereBro" ]]; then
            GPU_ID=$ind
            break
        fi
        ind=$[ind + 1]
    done
else
  echo 'Condition failed, probably not on HPC'
    # Not on HPC (probably)
    GPU_ID=$(nvidia-smi --list-gpus | awk '{NR;}END {print NR-1}') # Grabs number of GPUS
fi

if [ $USEGPU == 'true' ]; then
    echo "Using gpu: $GPU_ID"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    gpu=$GPU_ID
else
    echo "Using cpu"
    gpu=-1
fi


python train_classifier.py

