ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim graphviz \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev python-openslide wget libgl1-mesa-dev \
 && apt-get install -y language-pack-ja-base language-pack-ja \
 && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Install Miniconda
RUN wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# install tensorflow for faster-rcnn
COPY ./docker/requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm graphviz labelme==3.16.1 setuptools==45.2.0 tensorflow-gpu==1.12.0 pandas==1.0.5 scikit-learn==0.23.1
# install openslide which require setuptool==45.2.0.
RUN pip install openslide-python==1.1.1 

# Install PyTorch 1.1
ARG CUDA
RUN conda install pytorch torchvision cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision master
#RUN git clone https://github.com/pytorch/vision.git \
# && cd vision \
# && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && git checkout 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9 \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && git checkout ca00adace51ae4d82d2ff5cfc1ef1c9174ff6aaa \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Segmentation
WORKDIR /opt
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/sacmehta/ESPNet.git \
  && cd ESPNet \
  && git checkout afe71c38edaee3514ca44e0adcafdf36109bf437 \
  && cd ..

COPY ./module/espnet/train/* ./ESPNet/train/
COPY ./module/espnet/test/* ./ESPNet/test/
COPY ./module/common/*.py ./ESPNet/test/
COPY ./module/common/utils ./ESPNet/test/utils
COPY ./module/tools ./tools
COPY ./models ./ESPNet/models

# install programs related to faster-rcnn
RUN git clone https://github.com/jinseikenai/glomeruli_detection.git \
  && cd glomeruli_detection \
  && git checkout 4447a393a3cbbe68e090039867ed12e5a9e833eb \
  && cd ..

COPY ./module/faster-rcnn/* ./glomeruli_detection/
COPY ./module/common/*.py ./glomeruli_detection/
COPY ./module/common/utils ./glomeruli_detection/utils

# copy examples
COPY ./example ./ESPNet/example

WORKDIR /workspace
