#!/bin/bash

# Go root
sudo su

# Update the system and install Git
sudo apt update
sudo apt install -y git

# Set up Git configurations (replace with your details)
git config --global user.name "<username>"
git config --global user.email "<email>"

# Install Python 3 and pip
sudo apt install -y python3 python3-pip
python3 -m venv ccsd_env
source ccsd_env/bin/activate

# Install the required Python packages (with special cases)
pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install rdkit
pip install Cython
pip install pomegranate
pip install git+https://github.com/molecularsets/moses.git
pip install git+https://github.com/pyt-team/TopoNetX.git

# Clone the repository (replace with your repository URL)
git clone git@github.com:AdrienC21/CCSD.git

# Additionnal installations
sudo apt-get install libxrender1

# Navigate to the cloned repository directory
cd CCSD

# Apply fixes
python ./.github/workflows/apply_fixes.py
# Install the required Python packages (others)
pip install -r requirements.txt

# Finished!
echo "Post-installation script completed!"
echo "Executing tests ..."
pytest tests/ -W ignore::DeprecationWarning
echo "Test completed!"
