clear
echo "Started"
pip install -r requirements.txt
clear
echo "Installed all required packages"
wandb login 8a2ce1929f5f1d40c72093dc901f3cfe3e945b2e
clear
nvidia-smi
echo "Started Training"
python3 /home/groups/dustinms/aashrayc/glacformer-training/training/glacformer_training_script.py --continue-training $1 --num_epochs $2