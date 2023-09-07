#!/bin/bash

# Go root
sudo su

# Update the system and install Git and git-lfs (for MOSES)
echo "Update the system and install Git and git-lfs (for MOSES)"
sudo apt update
sudo apt install -y git
# git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo NEEDRESTART_MODE=a apt-get install git-lfs
git-lfs install

# Set up Git configurations (replace with your details)
echo "Set up Git configurations"
git config --global user.name ""
git config --global user.email ""

# Additionnal installations
echo "Additionnal installations"
sudo NEEDRESTART_MODE=a apt-get install -y libxss1
sudo NEEDRESTART_MODE=a apt-get install -y libxrender1
# sudo apt-get install xauth
# cp /home/ubuntu/.Xauthority /root/
# sed -i '$a AllowAgentForwarding yes' /etc/ssh/sshd_config
# sed -i '$a AllowTcpForwarding yes' /etc/ssh/sshd_config
# sed -i '$a X11DisplayOffset 10' /etc/ssh/sshd_config
# sed -i '$a X11UseLocalhost no' /etc/ssh/sshd_config
# sudo ufw allow ssh
# sudo ufw enable

# Install CUDA (https://hackmd.io/@MarconiJiang/nvidia_v100_ubuntu1804)
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
echo "Install CUDA"
distribution="ubuntu1804"
arch="x86_64"
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get -y update
sudo NEEDRESTART_MODE=a apt-get -y install cuda-drivers
sudo NEEDRESTART_MODE=a apt-get -y install cuda
sudo NEEDRESTART_MODE=a apt install -y nvidia-cuda-toolkit
cuda_version=cuda-12.1
# cuda_version=cuda-11.8
export PATH=/usr/local/$cuda_version/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/$cuda_version/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo apt-get install zlib1g
sudo NEEDRESTART_MODE=a apt-get install libcudnn8

# Install Python 3 and pip
echo "Install Python 3 and pip"
sudo NEEDRESTART_MODE=a apt install -y python3 python3-pip python3-venv
python3 -m venv ccsd_env
source ccsd_env/bin/activate

# Install the required Python packages (with special cases)
echo "Install the required Python packages (with special cases)"
pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install rdkit
pip install Cython
pip install pomegranate
git lfs install
pip install git+https://github.com/molecularsets/moses.git
pip install git+https://github.com/pyt-team/TopoNetX.git@a389bd8bb11c731bb98d79da8392e3396ea9db8c

# Install nodejs, npm, and orca for the plotly plots
sudo NEEDRESTART_MODE=a apt install -y nodejs
sudo NEEDRESTART_MODE=a apt install -y npm
npm install -g electron@6.1.4 orca

# Generate SSH keys
echo "Generate SSH keys"
ssh-keygen
echo "######################"
echo "Add the following SSH key to your GitHub account:"
cat ~/.ssh/id_rsa.pub

read -n 1 -s -r -p "Press any key to continue when the SSH key has been added to your GitHub account..."

# Clone the repository
echo "Clone the repository"
git clone git@github.com:AdrienC21/CCSD.git

# Navigate to the cloned repository directory
cd CCSD

# Apply fixes
echo "Apply fixes"
python ./.github/workflows/apply_fixes.py
# Compile orca
cd ccsd/src/evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
cd ../../../../
# Install the required Python packages (others)
echo "Install the required Python packages"
pip install -r requirements.txt
# Preprocess molecules datasets
echo "Preprocess molecules datasets"
python ccsd/data/preprocess.py --dataset QM9
python ccsd/data/preprocess_for_nspdk.py --dataset QM9
python ccsd/data/preprocess.py --dataset ZINC250k
python ccsd/data/preprocess_for_nspdk.py --dataset ZINC250k

# Add large files to save some time (OPTIONAL)
cd ../
git clone git@github.com:AdrienC21/temp_CCSD_files.git
cd CCSD
cp ../temp_CCSD_files/QM9_cc_True_train.pkl data/
cp ../temp_CCSD_files/QM9_cc_True_test.pkl data/
cp ../temp_CCSD_files/QM9_graphs_True_train.pkl data/
cp ../temp_CCSD_files/QM9_graphs_True_test.pkl data/

# Set up CUDA environment variables
export "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"

# Finished!
echo "Post-installation script completed!"
echo "Executing tests ..."
pytest tests/ -W ignore::DeprecationWarning
echo "Test completed! Reboot your system to finish the installation."
