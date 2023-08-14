#!/bin/bash

# Go root
sudo su

# Update the system and install Git and git-lfs (for MOSES)
sudo apt update
sudo apt install -y git
# git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git-lfs install

# Set up Git configurations (replace with your details)
git config --global user.name "<username>"
git config --global user.email "<email>"

# Install CUDA (https://hackmd.io/@MarconiJiang/nvidia_v100_ubuntu1804)
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
distribution="ubuntu1804"
arch="x86_64"
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get install cuda
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo apt-get install zlib1g
sudo apt-get install libcudnn8

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
git lfs install
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
