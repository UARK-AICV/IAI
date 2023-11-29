#/!/bin/bash
eval "$(conda shell.bash hook)" # copy conda command to shell
rm -rf ./env
conda create --prefix ./env python=3.8 -y
conda activate ./env
conda install -c conda-forge cxx-compiler==1.3.0 -y # to get g++ <= 10 for cuda 11.3
conda install  cudatoolkit=11.5 -c nvidia -y
# conda install -c conda-forge cudatoolkit-dev=11.3.1 -y
pip install nvidia-cudnn-cu11==8.6.0.163

export CUDA_HOME=''

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install mmcv-full # this may take ~20 mins, if this line failed, you should tru to reinstall `conda install -c conda-forge cudatoolkit-dev=11.3.1 -y`` again.
pip install fire
pip install transformers==4.30.2
pip install pandas
pip install SimpleITK
pip install einops torchmetrics