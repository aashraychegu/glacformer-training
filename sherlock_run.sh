#!/bin/bash

# Kill all existing tmux sessions
echo "Started"
tmux kill-server
ps aux | grep python
clear
module reset
module restore simsimd_install
pip install simsimd
source $HOME/.bash_profile
pip install -r requirements.txt
pip cache purge
echo "Installed all required packages"
nvidia-smi
echo "Started Training"
git config --global credential.helper store
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export script="/home/groups/dustinms/aashrayc/glacformer-training/training/glacformer_training_script.py"

# Default values
CONTINUE_TRAINING="False"
NUM_EPOCHS=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=1414
NUM_NODES=1
NODE_RANK=0
WANDB_ACTIVE=0
SESSION_NAME="GLACFORMER_TRAINING_SESSION_AASHRAYC"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --continue-training) CONTINUE_TRAINING="$2"; shift ;;
        --epochs) NUM_EPOCHS="$2"; shift ;;
        --master-addr) MASTER_ADDR="$2"; shift ;;
        --master-port) MASTER_PORT="$2"; shift ;;
        --num-nodes) NUM_NODES="$2"; shift ;;
        --node-rank) NODE_RANK="$2"; shift ;;
        --wandb-active) WANDB_ACTIVE="$2"; shift ;;
        --cleanup) TESTING="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$WANDB_ACTIVE" -eq 1 ]; then
    wandb login 8a2ce1929f5f1d40c72093dc901f3cfe3e945b2e
else
    wandb disabled
fi

# Start tmux session
#tmux new-session -d -s $SESSION_NAME

# Split the window into two panes vertically
#tmux split-window -h

# Run nvtop in the left pane
#tmux send-keys -t $SESSION_NAME:0.0 "nvtop" C-m

# Run the training script in the right pane
#tmux send-keys -t $SESSION_NAME:0.1 "torchrun --nnodes=$NUM_NODES --nproc-per-node=gpu --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT --node_rank=$NODE_RANK $script --continue_training $CONTINUE_TRAINING --num_epochs=$NUM_EPOCHS" C-m

# Attach to the tmux session
#tmux attach-session -t $SESSION_NAME

torchrun --nnodes=$NUM_NODES --nproc-per-node=gpu --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT --node_rank=$NODE_RANK $script --continue_training $CONTINUE_TRAINING --num_epochs=$NUM_EPOCHS

if [ "$TESTING" -eq "True" ]; then
    rm -rf ./glacformer
fi