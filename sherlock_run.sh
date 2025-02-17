#!/bin/bash


printf "\n\nStarted sherlock_run.sh\n\n"
ps aux | grep python
clear
module reset
module restore simsimd_install
pip install simsimd
module reset
source $HOME/.bash_profile
module restore default
pip install -r requirements.txt
pip install -U transformers[torch]
pip install -U albumentations
pip install -U accelerate
pip install -U wandb
pip cache purge
echo "Installed all required packages"
nvidia-smi
echo "Started Training"
git config --global credential.helper store

export script="/home/groups/dustinms/aashrayc/glacformer-training/training/glacformer_training_script.py"
module list

# Default values
CONTINUE_TRAINING="False"
NUM_EPOCHS=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=1414
NUM_NODES=1
WANDB_ACTIVE="False"
SESSION_NAME="GLACFORMER_TRAINING_SESSION_AASHRAYC"
TESTING="True"
CLEANUP="True"
JOBNAME="$SLURM_JOB_NAME-$SLURM_JOBID[$MASTER_NODE($SLURMD_NODENAME)-$SLURM_NODELIST]{QTIME:($(squeue -j $SLURM_JOB_ID -o "%V" -h))|STIME:($(date -d @$SLURM_JOB_START_TIME))|CTIME($(date))}@(Stanford Sherlock Cluster)"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --continue-training) CONTINUE_TRAINING="$2"; shift ;;
        --epochs) NUM_EPOCHS="$2"; shift ;;
        --master-addr) MASTER_ADDR="$2"; shift ;;
        --master-port) MASTER_PORT="$2"; shift ;;
        --num-nodes) NUM_NODES="$2"; shift ;;
        --wandb-active) WANDB_ACTIVE="$2"; shift ;;
        --cleanup) CLEANUP="$2"; shift ;;
        --testing) TESTING="$2"; shift ;;
        --jobname) JOBNAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$WANDB_ACTIVE" = "True" ]; then
    echo "USING WANDB"
    export WANDB_API_KEY=8a2ce1929f5f1d40c72093dc901f3cfe3e945b2e
else
    echo "WANDB DISABLED"
    export WANDB_MODE=disabled
fi
export WANDB_PROJECT="glacformer_training" 

export HF_TOKEN="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"
huggingface-cli whoami
huggingface-cli env

export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

if [ "$TESTING" = "False" ]; then
    torchrun --nnodes=$NUM_NODES --nproc-per-node=gpu --rdzv_id=GLACFORMER_TRAINING-$MASTER_ADDR-$MASTER_PORT --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $script --continue_training $CONTINUE_TRAINING --num_epochs=$NUM_EPOCHS --token $HF_TOKEN --jobname $JOBNAME
    EXIT_CODE=$?
    echo "EXIT_CODE: $EXIT_CODE"
else
    echo "STANDALONE TRAINING"
    ml ruse
    ruse torchrun --nnodes=1 --nproc-per-node=gpu --standalone $script --continue_training $CONTINUE_TRAINING --num_epochs $NUM_EPOCHS --token $HF_TOKEN --jobname "$JOBNAME"
    EXIT_CODE=$?
    echo "EXIT_CODE: $EXIT_CODE"
fi

if [ "$CLEANUP" = "True" ]; then
    rm -rf ./glacformer/*
fi

exit $EXIT_CODE