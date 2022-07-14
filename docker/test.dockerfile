ARG cuda_docker_tag="11.2.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:${cuda_docker_tag}

ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
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


