<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/logo-large.png" width="75%" />
</div>

![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen.svg)
[![PyPI](https://img.shields.io/pypi/v/torchopt?logo=pypi)](https://pypi.org/project/torchopt)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/metaopt/torchopt/Tests?label=tests&logo=github)
[![Documentation Status](https://img.shields.io/readthedocs/torchopt?logo=readthedocs)](https://torchopt.readthedocs.io)
[![Downloads](https://static.pepy.tech/personalized-badge/torchopt?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/torchopt)
[![GitHub Repo Stars](https://img.shields.io/github/stars/metaopt/torchopt?color=brightgreen&logo=github)](https://github.com/metaopt/torchopt/stargazers)
[![License](https://img.shields.io/github/license/metaopt/torchopt?label=license)](#license)

**TorchOpt** is an efficient library for differentiable optimization built upon [PyTorch](https://pytorch.org). TorchOpt is:

- **Comprehensive**: TorchOpt provides three differentiation mode - explicit differentiation, implicit differentiation and zero-order differentiation for handling different differenable optimization situations.
- **Flexible**: TorchOpt provides both functional and OOP API for user different preferences. Users can implement differentiable optimization in JAX-like or PyTorch-like style.
- **Efficient**: TorchOpt provides (1) CPU/GPU acceleration differentiable optimizer (2) RPC-based distributed training framework (3) Tree Operation parallelism, to largely increase the training efficiency for bi-level optimization problem.

In addition, torchopt can be served as

--------------------------------------------------------------------------------

The README is organized as follows:

- [TorchOpt for Differentiable topimization](#torchopt-for-differentiable-optimization)
  - [Explicit Gradient](#explicit-gradient)
  - [Implicit Gradient](#meta-learning-api)
  - [Zero-order Gradient](#meta-learning-api)
- [High-Performance and Distributed Training](#high-performance)
- [Visualization](#visualization)
- [Examples](#examples)
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

We also provide a wrapper `torchopt.FuncOptimizer` to make maintaining the optimizer state easier:

```python
net = Net()  # init
loader = Loader()
optimizer = torchopt.FuncOptimizer(torchopt.adam())      # wrap with `torchopt.FuncOptimizer`

model, params = functorch.make_functional(net)           # use functorch extract network parameters

for xs, ys in loader:                                    # get data
    pred = model(params, xs)                             # forward
    loss = F.cross_entropy(pred, ys)                     # compute loss

    params = optimizer.step(loss, params)                # update network parameters
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

On top of the same optimization function as `torch.optim`, an important benefit of functional optimizer is that one can implement differentiable optimization easily. This is particularly helpful when the algorithm requires to differentiate through optimization update (such as meta learning practices). We take as the inputs the gradients and optimizer states, use non-in-place operators to compute and output the updates. The processes can be automatically implemented, with the only need from users being to pass the argument `inplace=False` to the functions. You canCheck [Explicit Gradient](#explicit-gradient) functional API for example.

--------------------------------------------------------------------------------

## TorchOpt for Differentiable Optimization

Meta-Learning has gained enormous attention in both Supervised Learning and Reinforcement Learning. Meta-Learning algorithms often contain a bi-level optimization process with *inner loop* updating the network parameters and *outer loop* updating meta parameters. The figure below illustrates the basic formulation for meta-optimization in Meta-Learning. The main feature is that the gradients of *outer loss* will back-propagate through all `inner.step` operations.

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/TorchOpt.png" width="85%" />
</div>

Since network parameters become a node of computation graph, a flexible Meta-Learning library should enable users manually control the gradient graph connection which means that users should have access to the network parameters and optimizer states for manually detaching or connecting the computation graph. In PyTorch designing, the network parameters or optimizer states are members of network (a.k.a. `torch.nn.Module`) or optimizer (a.k.a. `torch.optim.Optimizer`), this design significantly introducing difficulty for user control network parameters or optimizer states. Previous differentiable optimizer Repo [`higher`](https://github.com/facebookresearch/higher), [`learn2learn`](https://github.com/learnables/learn2learn) follows the PyTorch designing which leads to inflexible API.

In contrast to them, TorchOpt realizes differentiable optimizer with functional programing, where Meta-Learning researchers could control the network parameters or optimizer states as normal variables (a.k.a. `torch.Tensor`). This functional optimizer design of TorchOpt is beneficial for implementing complex gradient flow Meta-Learning algorithms and allow us to improve computational efficiency by using techniques like operator fusion.

### Explicit Gradient
#### Functional API
```python
# Functional API
optimizer = torchopt.adam()
# Define meta and inner parameters
meta_params = ...
fmodel, params = make_functional(model)
# Initial state
state = optimizer.init(params)

for iter in range(iter_times):
    loss = inner_loss(fmodel, params, meta_params)                
    grads = torch.autograd.grad(loss, params)
    # Apply non-inplace parameter update
    updates, state = optimizer.update(grads, state, inplace=False)
    params = torchopt.apply_updates(params, updates)   
    
loss = outer_loss(fmodel, params, meta_p)
meta_grads = torch.autograd.grad(loss, meta_params)
```
functional optimizer which enables [JAX-like](https://github.com/google/jax) composable functional optimizer for PyTorch. With TorchOpt, one can easily conduct neural network optimization in PyTorch with functional style optimizer, similar to  [Optax](https://github.com/deepmind/optax) in JAX.
### OOP API
```python
# OOP API
# Define meta and inner parameters
meta_params = ...
model = ...
# Define differentiable optimizer
optimizer = torchopt.MetaAdam(model.parameters())

for iter in range(iter_times):
    # perform inner update
    loss = inner_loss(model, meta_params)  
    optimizer.step(loss)
    
loss = outer_loss(model, meta_p)
loss.backward()
```
- We design a base class `torchopt.MetaOptimizer` for managing network updates in Meta-Learning. The constructor of `MetaOptimizer` takes as input the network rather than network parameters. `MetaOptimizer` exposed interface `step(loss)` takes as input the loss for step the network parameter. Refer to the tutorial notebook [Meta Optimizer](tutorials/3_Meta_Optimizer.ipynb) for more details.
- We offer `torchopt.chain` which can apply a list of chainable update transformations. Combined with `MetaOptimizer`, it can help you conduct gradient transformation such as gradient clip before the Meta optimizer steps. Refer to the tutorial notebook [Meta Optimizer](tutorials/3_Meta_Optimizer.ipynb) for more details.
- We observe that different Meta-Learning algorithms vary in inner-loop parameter recovery. TorchOpt provides basic functions for users to extract or recover network parameters and optimizer states anytime anywhere they want.
- Some algorithms such as MGRL ([arXiv:1805.09801](https://arxiv.org/abs/1805.09801)) initialize the inner-loop parameters inherited from previous inner-loop process when conducting a new bi-level process. TorchOpt also provides a finer function `stop_gradient` for manipulating the gradient graph, which is helpful for this kind of algorithms. Refer to the notebook [Stop Gradient](tutorials/4_Stop_Gradient.ipynb) for more details.

### Implicit Gradient
#### Functional API
```python
# Functional API for implicit gradient
def stationary(params, meta_params, data):
    # stationary condition construction
    return stationary condition

@torchopt.implicit.custom_root(stationary)
def solve(params, meta_params, data):
    # forward optimization process
    return output
    
# Define params, meta params and get data
params, meta_prams, data = ..., ..., ...
optimal_params = solve(params, meta_params, data)
loss = outer_loss(optimal_params)

meta_grads = torch.autograd.grad(loss, meta_params)
```
#### OOP API

### Zero-order gradient
#### Functional API
```python
# Functional API
# Specify method and parameter of ES
@torchopt.zero_order.es_grad(method, params)
def forward(meta_params, data):
    # forward process
    return output
```
--------------------------------------------------------------------------------
## High-Performance

One can think of the scale procedures on gradients of optimizer algorithms as a combination of several operations. For example, the implementation of the Adam algorithm often includes addition, multiplication, power and square operations, one can fuse these operations into several compound functions. The operator fusion could greatly simplify the computation graph and reduce the GPU function launching stall. In addition, one can also implement the optimizer backward function and manually reuse some intermediate tensors to improve the backward performance. Users can pass argument `use_accelerated_op=True` to `adam`, `Adam` and `MetaAdam` to enable the fused accelerated operator. The arguments are the same between the two kinds of implementations.

Here we evaluate the performance using the MAML-Omniglot code with the inner-loop Adam optimizer on GPU. We comparable the run time of the overall algorithm and the meta-optimization (outer-loop optimization) under different network architecture/inner-step numbers. We choose [`higher`](https://github.com/facebookresearch/higher) as our baseline. The figure below illustrate that our accelerated Adam can achieve at least $1/3$ efficiency improvement over the baseline.

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/time.png" width="80%" />
</div>

Notably, the operator fusion not only increases performance but also help simplify the computation graph, which will be discussed in the next section.

--------------------------------------------------------------------------------

## Visualization

Complex gradient flow in meta-learning brings in a great challenge for managing the gradient flow and verifying the correctness of it. TorchOpt provides a visualization tool that draw variable (e.g. network parameters or meta parameters) names on the gradient graph for better analyzing. The visualization tool is modified from [`torchviz`](https://github.com/szagoruyko/pytorchviz). We provide an example using the [visualization code](examples/visualize.py). Also refer to the notebook [Visualization](tutorials/2_Visualization.ipynb) for more details.

The figure below show the visualization result. Compared with [`torchviz`](https://github.com/szagoruyko/pytorchviz), TorchOpt fuses the operations within the `Adam` together (orange) to reduce the complexity and provide simpler visualization.

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/torchviz_torchopt.jpg" width="80%" />
</div>

--------------------------------------------------------------------------------
## Examples

In [`examples`](examples), we offer several examples of functional optimizer and 5 light-weight meta-learning examples with TorchOpt. The meta-learning examples covers 2 Supervised Learning and 3 Reinforcement Learning algorithms.

- [Model Agnostic Meta Learning (MAML) - Supervised Learning](https://arxiv.org/abs/1703.03400) (ICML 2017)
- [Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/abs/1803.09050) (ICML 2018)
- [Model Agnostic Meta Learning (MAML) - Reinforcement Learning](https://arxiv.org/abs/1703.03400) (ICML 2017)
- [Meta Gradient Reinforcement Learning (MGRL)](https://arxiv.org/abs/1805.09801) (NeurIPS 2018)
- [Learning through opponent learning process (LOLA)](https://arxiv.org/abs/1709.04326) (AAMAS 2018)

--------------------------------------------------------------------------------


## Installation

Requirements

- PyTorch
- (Optional) For visualizing computation graphs
  - [Graphviz](https://graphviz.org/download) (for Linux users use `apt/yum install graphviz` or `conda install -c anaconda python-graphviz`)

**Please follow the instructions at <https://pytorch.org> to install PyTorch in your Python environment first.** Then run the following command to install TorchOpt from PyPI ([![PyPI](https://img.shields.io/pypi/v/torchopt?label=pypi&logo=pypi)](https://pypi.org/project/torchopt) / ![Status](https://img.shields.io/pypi/status/torchopt?label=status)):

```bash
pip3 install torchopt
```

If the minimum version of PyTorch is not satisfied, `pip` will install/upgrade it for you. Please be careful about the `torch` build for CPU / CUDA support (e.g. `cpu`, `cu116`, `cu117`). You may need to specify the extra index URL for the `torch` package:

```bash
pip3 install torchopt --extra-index-url https://download.pytorch.org/whl/cu117
```

See <https://pytorch.org> for more information about installing PyTorch.

You can also build shared libraries from source, use:

```bash
git clone https://github.com/metaopt/torchopt.git
cd torchopt
pip3 install .
```

We provide a [conda](https://github.com/conda/conda) environment recipe to install the build toolchain such as `cmake`, `g++`, and `nvcc`:

```bash
git clone https://github.com/metaopt/torchopt.git
cd torchopt

# You may need `CONDA_OVERRIDE_CUDA` if conda fails to detect the NVIDIA driver (e.g. in docker or WSL2)
CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda-recipe.yaml

conda activate torchopt
make install-editable  # or run `pip3 install --no-build-isolation --editable .`
```

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

--------------------------------------------------------------------------------

## The Team

TorchOpt is a work by Jie Ren, Xidong Feng(https://github.com/waterhorse1), [Bo Liu](https://github.com/Benjamin-eecs), [Xuehai Pan](https://github.com/XuehaiPan), [Luo Mai](https://luomai.github.io/) and [Yaodong Yang](https://www.yangyaodong.com/).

## Citing TorchOpt

If you find TorchOpt useful, please cite it in your publications.

```bibtex
@software{TorchOpt,
  author       = {Jie Ren and Xidong Feng and Bo Liu and Xuehai Pan and Luo Mai and Yaodong Yang},
  title        = {TorchOpt},
  year         = {2022},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/metaopt/torchopt}}
}
```
