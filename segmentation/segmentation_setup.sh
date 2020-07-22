#!/bin/bash

sudo su

apt update
apt install software-properties-common

add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.6

update-alternatives --config python
update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
update-alternatives --config python

curl https://bootstrap.pypa.io/get-pip.py | python - --user

cp /home/ubuntu/.local/bin/pip /usr/bin/
cp /home/ubuntu/.local/bin/pip3 /usr/bin/

pip install opencv-contrib-python==3.4.4.19
pip install numpy==1.14.5

cd ~
apt install wget
wget https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz
tar -zxf git-lfs-linux-amd64-2.3.4.tar.gz
cd git-lfs-2.3.4
./install.sh

cd ~
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
mv ./cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64 ./cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
apt-get update
apt-get install cuda-9-2

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

wget https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torch-0.4.1-cp36-cp36m-linux_x86_64.whl

pip install pytorch-ignite==0.1.0
pip install pillow
pip install tqdm
pip install visdom
