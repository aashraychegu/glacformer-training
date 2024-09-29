clear
echo "Started"
pip install -r requirements.txt
clear
echo "Installed all required packages"
wandb login 8a2ce1929f5f1d40c72093dc901f3cfe3e945b2e
clear
echo "Started Training"
python3 training/glacformer_training_script.py --load_from new --num_epochs 15