#!/bin/bash

apt-get -y install git
apt-get -y install gcc
apt-get -y install make
apt-get -y install g++
apt-get -y install python-dev
apt-get -y install python-tk
apt-get -y install unzip
apt-get -y install htop
apt-get -y install nmap
apt-get -y install libcupti-dev

#git lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v2.1.0/git-lfs-linux-amd64-2.1.0.tar.gz
tar -zxvf git-lfs-linux-amd64-2.1.0.tar.gz
cd git-lfs-2.1.0/
bash install.sh
cd ..

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
  apt-get install linux-headers-$(uname -r) -y
fi

#gcloud compute copy-files ~/Desktop/libcudnn5_5.1.10-1+cuda8.0_amd64.deb kg1:/home/dd_petrovskiy/tmp
#sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install scipy
pip install pandas
pip install -U scikit-learn
pip install hyperopt
pip install dill
pip install seaborn
pip install matplotlib
pip install gensim
pip install tensorflow-gpu
pip install keras
pip install -U nltk
pip install h5py
pip install distance
pip install eli5
pip install tqdm
pip install fuzzywuzzy
pip install python-levenshtein
pip install jupyter
pip install IPython
pip install dask[complete]

pip install -U spacy
python -m spacy download en


git clone http://github.com/dmlc/xgboost
cd xgboost
git submodule update --init
./build.sh
cd python-package
python setup.py install
