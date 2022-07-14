#!/bin/bash

CPU_PARENT=ubuntu:18.04
GPU_PARENT=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

TAG=metaopt/TorchOpt
VERSION=$(shell git log -1 --format=%h)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  PYTORCH_DEPS="cudatoolkit=10.1"
else
  PARENT=${CPU_PARENT}
  PYTORCH_DEPS="cpuonly"
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.16 clang-format-11 git wget \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN pip3 install --upgrade pip isort yapf cpplint flake8 flake8_bugbear mypy && rm -rf ~/.pip/cache

WORKDIR /app
COPY . .



ARG cuda_docker_tag="11.2.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:${cuda_docker_tag}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    tzdata \
    wget   \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    gcc \
    htop\
    git \
    tmux \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV LANG C.UTF-8
ENV CONDA_DIR /opt/conda
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN mkdir -p /root/T \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \


# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
COPY . /home
RUN conda clean --all --force-pkgs-dirs --yes
RUN eval "$(conda shell.bash hook)" && \
   conda activate base              && \
   python -m pip install --upgrade pip &&\
   python -m pip install -e /home/. &&\
   rm -rf ~/.pip/cache
WORKDIR /home



sudo docker run -v /home/liubo/_liubo/Projects/high_performance_simulator/yufan/envpool/:/home/ -it --env HTTP_PROXY=HTTPS_PROXY="http://127.0.0.1:9910" envpool:949c6fe