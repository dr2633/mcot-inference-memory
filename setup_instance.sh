#!/bin/bash

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python and dependencies..."
sudo apt install -y python3 python3-pip git

echo "Setting up virtual environment..."
python3 -m venv ~/venv
source ~/venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch with GPU support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Cloning your repo..."
git clone https://github.com/dr2633/mcot-inference-memory ~/mCoT-GRPO-IPO
cd ~/mCoT-GRPO-IPO

echo "Installing project dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Verifying installation..."
python3 -c "import torch, transformers, datasets; print('Dependencies installed!')"

echo "Running benchmark tests..."
python3 cot/run_baseline_cot.py --subset_size 50 --temperature 0.7

echo "Setup complete! Your instance is ready to use."