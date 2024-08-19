#conda create --name mmseg python=3.8 -y
#conda activate mmseg

conda install cudatoolkit=11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install -v -e .
