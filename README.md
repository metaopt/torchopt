<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="image/logo-large.png" width="75%" />
</div>

![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen.svg)
[![PyPI](https://img.shields.io/pypi/v/torchopt?label=PyPI)](https://pypi.org/project/torchopt)
![Status](https://img.shields.io/pypi/status/torchopt?label=Status)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/metaopt/TorchOpt/Tests?label=tests&logo=github)
[![Documentation Status](https://readthedocs.org/projects/torchopt/badge/?version=latest)](https://torchopt.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/personalized-badge/torchopt?period=month&left_color=grey&right_color=blue&left_text=Downloads/month)](https://pepy.tech/project/torchopt)
[![GitHub Repo Stars](https://img.shields.io/github/stars/metaopt/torchopt?label=Stars&logo=github&color=brightgreen)](https://github.com/metaopt/torchopt/stargazers)
[![License](https://img.shields.io/github/license/metaopt/TorchOpt?label=License)](#license)

**TorchOpt** is a high-performance optimizer library built upon [PyTorch](https://pytorch.org/) for easy implementation of functional optimization and gradient-based meta-learning. It consists of two main features:

- TorchOpt provides functional optimizer which enables [JAX-like](https://github.com/google/jax) composable functional optimizer for PyTorch. With TorchOpt, one can easily conduct neural network optimization in PyTorch with functional style optimizer, similar to  [Optax](https://github.com/deepmind/optax) in JAX.
- With the design of functional programing, TorchOpt provides efficient, flexible, and easy-to-implement differentiable optimizer for gradient-based meta-learning research. It largely reduces the efforts required to implement sophisticated meta-learning algorithms.

--------------------------------------------------------------------------------

The README is organized as follows:

- [TorchOpt as Functional Optimizer](#torchopt-as-functional-optimizer)
  - [Optax-Like API](#optax-like-api)
  - [PyTorch-Like API](#pytorch-like-api)
  - [Differentiable](#differentiable)
- [TorchOpt as Differentiable Optimizer for Meta-Learning](#torchopt-as-differentiable-optimizer-for-meta-learning)
  - [Meta-Learning API](#meta-learning-api)
- [Examples](#examples)
- [High-Performance](#high-performance)
- [Visualization](#visualization)
- [Installation](#installation)
- [Future Plan](#future-plan)
- [Changelog](#changelog)
- [The Team](#the-team)
- [Citing TorchOpt](#citing-torchopt)

--------------------------------------------------------------------------------

## TorchOpt as Functional Optimizer

The design of TorchOpt follows the philosophy of functional programming. Aligned with [`functorch`](https://github.com/pytorch/functorch), users can conduct functional style programing with models, optimizers and training in PyTorch. We use the Adam optimizer as an example in the following illustration. You can also check out the tutorial notebook [Functional Optimizer](tutorials/1_Functional_Optimizer.ipynb) for more details.

### Optax-Like API

For those users who prefer fully functional programing, we offer Optax-Like API by passing gradients and optimizers states to the optimizer function. We design base class `torchopt.Optimizer` that has the same interface as `torch.optim.Optimizer`. Here is an example coupled with `functorch`:

```python
import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchopt

class Net(nn.Module): ...

class Loader(DataLoader): ...

net = Net()  # init
loader = Loader()
optimizer = torchopt.adam()

model, params = functorch.make_functional(net)           # use functorch extract network parameters
opt_state = optimizer.init(params)                       # init optimizer

xs, ys = next(loader)                                    # get data
pred = model(params, xs)                                 # forward
loss = F.cross_entropy(pred, ys)                         # compute loss

grads = torch.autograd.grad(loss, params)                # compute gradients
updates, opt_state = optimizer.update(grads, opt_state)  # get updates
params = torchopt.apply_updates(params, updates)         # update network parameters
```

### PyTorch-Like API

We also offer origin PyTorch APIs (e.g. `zero_grad()` or `step()`) by wrapping our Optax-Like API for traditional PyTorch user:

```python
net = Net()  # init
loader = Loader()
optimizer = torchopt.Adam(net.parameters())

xs, ys = next(loader)             # get data
pred = net(xs)                    # forward
loss = F.cross_entropy(pred, ys)  # compute loss

optimizer.zero_grad()             # zero gradients
loss.backward()                   # backward
optimizer.step()                  # step updates
```

### Differentiable

On top of the same optimization function as `torch.optim`, an important benefit of functional optimizer is that one can implement differentiable optimization easily. This is particularly helpful when the algorithm requires to differentiate through optimization update (such as meta learning practices). We take as the inputs the gradients and optimizer states, use non-in-place operators to compute and output the updates. The processes can be automatically implemented, with the only need from users being to pass the argument `inplace=False` to the functions:

```python
# Get updates
updates, opt_state = optimizer.update(grad, opt_state, inplace=False)
# Update network parameters
params = torchopt.apply_updates(params, updates, inplace=False)
```

--------------------------------------------------------------------------------

## TorchOpt as Differentiable Optimizer for Meta-Learning

Meta-Learning has gained enormous attention in both Supervised Learning and Reinforcement Learning. Meta-Learning algorithms often contain a bi-level optimization process with *inner loop* updating the network parameters and *outer loop* updating meta parameters. The figure below illustrates the basic formulation for meta-optimization in Meta-Learning. The main feature is that the gradients of *outer loss* will back-propagate through all `inner.step` operations.

<div align="center">
  <img src="/image/TorchOpt.png" width="85%" />
</div>

Since network parameters become a node of computation graph, a flexible Meta-Learning library should enable users manually control the gradient graph connection which means that users should have access to the network parameters and optimizer states for manually detaching or connecting the computation graph. In PyTorch designing, the network parameters or optimizer states are members of network (a.k.a. `torch.nn.Module`) or optimizer (a.k.a. `torch.optim.Optimizer`), this design significantly introducing difficulty for user control network parameters or optimizer states. Previous differentiable optimizer Repo [`higher`](https://github.com/facebookresearch/higher), [`learn2learn`](https://github.com/learnables/learn2learn) follows the PyTorch designing which leads to inflexible API.

In contrast to them, TorchOpt realizes differentiable optimizer with functional programing, where Meta-Learning researchers could control the network parameters or optimizer states as normal variables (a.k.a. `torch.Tensor`). This functional optimizer design of TorchOpt is beneficial for implementing complex gradient flow Meta-Learning algorithms and allow us to improve computational efficiency by using techniques like operator fusion.

### Meta-Learning API

- We design a base class `torchopt.MetaOptimizer` for managing network updates in Meta-Learning. The constructor of `MetaOptimizer` takes as input the network rather than network parameters. `MetaOptimizer` exposed interface `step(loss)` takes as input the loss for step the network parameter. Refer to the tutorial notebook [Meta Optimizer](tutorials/3_Meta_Optimizer.ipynb) for more details.
- We offer `torchopt.chain` which can apply a list of chainable update transformations. Combined with `MetaOptimizer`, it can help you conduct gradient transformation such as gradient clip before the Meta optimizer steps. Refer to the tutorial notebook [Meta Optimizer](tutorials/3_Meta_Optimizer.ipynb) for more details.
- We observe that different Meta-Learning algorithms vary in inner-loop parameter recovery. TorchOpt provides basic functions for users to extract or recover network parameters and optimizer states anytime anywhere they want.
- Some algorithms such as MGRL ([arXiv:1805.09801](https://arxiv.org/abs/1805.09801)) initialize the inner-loop parameters inherited from previous inner-loop process when conducting a new bi-level process. TorchOpt also provides a finer function `stop_gradient` for manipulating the gradient graph, which is helpful for this kind of algorithms. Refer to the notebook [Stop Gradient](tutorials/4_Stop_Gradient.ipynb) for more details.

We give an example of MAML ([arXiv:1703.03400](https://arxiv.org/abs/1703.03400)) with inner-loop Adam optimizer to illustrate TorchOpt APIs:

```python
net = Net()  # init

# The constructor `MetaOptimizer` takes as input the network
inner_optim = torchopt.MetaAdam(net)
outer_optim = torchopt.Adam(net.parameters())

for train_iter in range(train_iters):
    outer_loss = 0
    for task in range(tasks):
        loader = Loader(tasks)

        # Store states at the initial points
        net_state = torchopt.extract_state_dict(net)  # extract state
        optim_state = torchopt.extract_state_dict(inner_optim)
        for inner_iter in range(inner_iters):
            # Compute inner loss and perform inner update
            xs, ys = next(loader)
            pred = net(xs)
            inner_loss = F.cross_entropy(pred, ys)
            inner_optim.step(inner_loss)

        # Compute outer loss and back-propagate
        xs, ys = next(loader)
        pred = net(xs)
        outer_loss = outer_loss + F.cross_entropy(pred, ys)

        # Recover network and optimizer states at the initial point for the next task
        torchopt.recover_state_dict(inner_optim, optim_state)
        torchopt.recover_state_dict(net, net_state)

    outer_loss = outer_loss / len(tasks)  # task average
    outer_optim.zero_grad()
    outer_loss.backward()
    outer_optim.step()

    # Stop gradient if necessary
    torchopt.stop_gradient(net)
    torchopt.stop_gradient(inner_optim)
```

--------------------------------------------------------------------------------

## Examples

In [`examples`](examples), we offer several examples of functional optimizer and 5 light-weight meta-learning examples with TorchOpt. The meta-learning examples covers 2 Supervised Learning and 3 Reinforcement Learning algorithms.

- [Model Agnostic Meta Learning (MAML) - Supervised Learning](https://arxiv.org/abs/1703.03400) (ICML2017)
- [Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/abs/1803.09050) (ICML2018)
- [Model Agnostic Meta Learning (MAML) - Reinforcement Learning](https://arxiv.org/abs/1703.03400) (ICML2017)
- [Meta Gradient Reinforcement Learning (MGRL)](https://arxiv.org/abs/1805.09801) (NeurIPS 2018)
- [Learning through opponent learning process (LOLA)](https://arxiv.org/abs/1709.04326) (AAMAS 2018)

--------------------------------------------------------------------------------

## High-Performance

One can think of the scale procedures on gradients of optimizer algorithms as a combination of several operations. For example, the implementation of the Adam algorithm often includes addition, multiplication, power and square operations, one can fuse these operations into several compound functions. The operator fusion could greatly simplify the computation graph and reduce the GPU function launching stall. In addition, one can also implement the optimizer backward function and manually reuse some intermediate tensors to improve the backward performance. Users can pass argument `use_accelerated_op=True` to `adam`, `Adam` and `MetaAdam` to enable the fused accelerated operator. The arguments are the same between the two kinds of implementations.

Here we evaluate the performance using the MAML-Omniglot code with the inner-loop Adam optimizer on GPU. We comparable the run time of the overall algorithm and the meta-optimization (outer-loop optimization) under different network architecture/inner-step numbers. We choose [`higher`](https://github.com/facebookresearch/higher) as our baseline. The figure below illustrate that our accelerated Adam can achieve at least $1/3$ efficiency improvement over the baseline.

<div align="center">
  <img src="image/time.png" width="80%" />
</div>

Notably, the operator fusion not only increases performance but also help simplify the computation graph, which will be discussed in the next section.

--------------------------------------------------------------------------------

## Visualization

Complex gradient flow in meta-learning brings in a great challenge for managing the gradient flow and verifying the correctness of it. TorchOpt provides a visualization tool that draw variable (e.g. network parameters or meta parameters) names on the gradient graph for better analyzing. The visualization tool is modified from [`torchviz`](https://github.com/szagoruyko/pytorchviz). We provide an example using the [visualization code](examples/visualize.py). Also refer to the notebook [Visualization](tutorials/2_Visualization.ipynb) for more details.

The figure below show the visualization result. Compared with [`torchviz`](https://github.com/szagoruyko/pytorchviz), TorchOpt fuses the operations within the `Adam` together (orange) to reduce the complexity and provide simpler visualization.

<div align="center">
  <img src="image/torchviz_torchopt.jpg" width="80%" />
</div>

--------------------------------------------------------------------------------

## Installation

Requirements

- PyTorch
- JAX
- (Optional) For visualizing computation graphs
  - [Graphviz](https://graphviz.org/download/) (for Linux users use `apt/yum install graphviz` or `conda install -c anaconda python-graphviz`)

Please follow the instructions at <https://pytorch.org> to install PyTorch in your Python environment first. Then run the following command to install TorchOpt from PyPI ([![PyPI](https://img.shields.io/pypi/v/torchopt?label=PyPI)](https://pypi.org/project/torchopt) / ![Status](https://img.shields.io/pypi/status/torchopt?label=Status)):

```bash
pip3 install torchopt
```

You can also build shared libraries from source, use:

```bash
git clone https://github.com/metaopt/TorchOpt.git
cd TorchOpt
pip3 install .
```

We provide a [conda](https://github.com/conda/conda) environment recipe to install the build toolchain such as `cmake`, `g++`, and `nvcc`:

```bash
git clone https://github.com/metaopt/TorchOpt.git
cd TorchOpt

# You may need `CONDA_OVERRIDE_CUDA` if conda fails to detect the NVIDIA driver (e.g. in docker or WSL2)
CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda-recipe.yaml

conda activate torchopt
pip3 install --no-build-isolation --editable .
```

--------------------------------------------------------------------------------

## Future Plan

- [ ] Support general implicit differentiation with functional programing.
- [ ] Support more optimizers such as AdamW, RMSProp
- [ ] CPU-accelerated optimizer

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

--------------------------------------------------------------------------------

## The Team

TorchOpt is a work by Jie Ren, Xidong Feng, [Bo Liu](https://github.com/Benjamin-eecs), [Xuehai Pan](https://github.com/XuehaiPan), [Luo Mai](https://luomai.github.io/) and [Yaodong Yang](https://www.yangyaodong.com/).

## Citing TorchOpt

If you find TorchOpt useful, please cite it in your publications.

```bibtex
@software{TorchOpt,
  author = {Jie Ren and Xidong Feng and Bo Liu and Xuehai Pan and Luo Mai and Yaodong Yang},
  title = {TorchOpt},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/metaopt/TorchOpt}},
}
```
