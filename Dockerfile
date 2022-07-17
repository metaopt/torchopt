# Dockerfile for TorchOpt
#
#   $ docker build --target base --tag torchopt:latest .
#
# or
#
#   $ docker build --target devel --tag torchopt-devel:latest .
#

ARG cuda_docker_tag="11.6.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:"${cuda_docker_tag}" AS builder

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Install packages
RUN apt-get update && \
    apt-get install -y sudo ca-certificates openssl \
        git ssh build-essential gcc-10 g++-10 cmake make \
        python3.9-dev python3.9-venv graphviz && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV CC=gcc-10 CXX=g++-10

# Add a new user
RUN useradd -m -s /bin/bash torchopt && \
    echo "torchopt ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER torchopt
RUN echo "export PS1='[\[\e[1;33m\]\u\[\e[0m\]:\[\e[1;35m\]\w\[\e[0m\]]\$ '" >> ~/.bashrc

# Setup virtual environment
RUN /usr/bin/python3.9 -m venv --upgrade-deps ~/venv && rm -rf ~/.pip/cache
RUN TORCH_INDEX_URL="https://download.pytorch.org/whl/cu$(echo "${CUDA_VERSION}" | cut -d'.' -f-2  | tr -d '.')" && \
    echo "export TORCH_INDEX_URL='${TORCH_INDEX_URL}'" >> ~/venv/bin/activate && \
    echo "source /home/torchopt/venv/bin/activate" >> ~/.bashrc

# Install dependencies
WORKDIR /home/torchopt/TorchOpt
COPY --chown=torchopt requirements.txt requirements.txt
RUN source ~/venv/bin/activate && \
    python -m pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt && \
    rm -rf ~/.pip/cache ~/.cache/pip

####################################################################################################

FROM builder AS devel-builder

# Install extra dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y golang-1.16 clang-format clang-tidy && \
    sudo chown -R "$(whoami):$(whoami)" /usr/lib/go-1.16 && \
    sudo rm -rf /var/lib/apt/lists/*

# Install addlicense
ENV GOPATH="/usr/lib/go-1.16"
ENV GOBIN="${GOPATH}/bin"
ENV GOROOT="${GOPATH}"
ENV PATH="${GOBIN}:${PATH}"
RUN go install github.com/google/addlicense@latest

# Install extra PyPI dependencies
COPY --chown=torchopt tests/requirements.txt tests/requirements.txt
RUN source ~/venv/bin/activate && \
    python -m pip install --extra-index-url "${TORCH_INDEX_URL}" -r tests/requirements.txt && \
    rm -rf ~/.pip/cache ~/.cache/pip

####################################################################################################

FROM builder AS base

COPY --chown=torchopt . .

# Install TorchOpt
RUN source ~/venv/bin/activate && \
    python -m pip install -e . && \
    rm -rf .eggs *.egg-info ~/.pip/cache ~/.cache/pip

ENTRYPOINT [ "/bin/bash", "--login" ]

####################################################################################################

FROM devel-builder AS devel

COPY --from=base /home/torchopt/TorchOpt .
