
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker


# Sane Defaults
RUN apt-get update
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    libgl1 \
    libopenexr-dev \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame


# IMPORTANT :: Uninstall & Reinstall Torch (Sagemaker CPU Core Bug)
RUN pip install --upgrade pip

RUN pip uninstall -y torch
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip uninstall -y torch
RUN pip install pytorch_lightning==1.9.3
RUN pip install xformers==0.0.24
RUN pip uninstall -y torch torchvision
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install fairscale


RUN pip install sagemaker-training

RUN pip install ipdb

ENV PATH="/opt/ml/code:${PATH}"
ENV PATH="/opt/ml/code/DynamiCrafter:${PATH}"


ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/main/trainer.py

COPY . /opt/ml/code/

RUN cp /opt/ml/code/.bashrc ~/.bashrc
RUN cp /opt/ml/code/.vimrc ~/.vimrc


