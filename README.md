<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/logo-large.png" width="75%" />
</div>

<div align="center">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  <a href="https://pypi.org/project/torchopt">![PyPI](https://img.shields.io/pypi/v/torchopt?logo=pypi)</a>
  <a href="https://github.com/metaopt/torchopt/tree/HEAD/tests">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/metaopt/torchopt/tests.yml?label=tests&logo=github)</a>
  <a href="https://codecov.io/gh/metaopt/torchopt">![CodeCov](https://img.shields.io/codecov/c/github/metaopt/torchopt/main?logo=codecov)</a>
  <a href="https://torchopt.readthedocs.io">![Documentation Status](https://img.shields.io/readthedocs/torchopt?logo=readthedocs)</a>
  <a href="https://pepy.tech/project/torchopt">![Downloads](https://static.pepy.tech/personalized-badge/torchopt?period=total&left_color=grey&right_color=blue&left_text=downloads)</a>
  <a href="https://github.com/metaopt/torchopt/blob/HEAD/LICENSE">![License](https://img.shields.io/github/license/metaopt/torchopt?label=license&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)</a>
</div>

<p align="center">
  <a href="https://github.com/metaopt/torchopt#installation">Installation</a> |
  <a href="https://torchopt.readthedocs.io">Documentation</a> |
  <a href="https://github.com/metaopt/torchopt/tree/HEAD/tutorials">Tutorials</a> |
  <a href="https://github.com/metaopt/torchopt/tree/HEAD/examples">Examples</a> |
  <a href="https://arxiv.org/abs/2211.06934">Paper</a> |
  <a href="https://github.com/metaopt/torchopt#citing-torchopt">Citation</a>
</p>

**TorchOpt** is an efficient library for differentiable optimization built upon [PyTorch](https://pytorch.org).
TorchOpt is:

- **Comprehensive**: TorchOpt provides three differentiation modes - explicit differentiation, implicit differentiation, and zero-order differentiation for handling different differentiable optimization situations.
- **Flexible**: TorchOpt provides both functional and objective-oriented API for users' different preferences. Users can implement differentiable optimization in JAX-like or PyTorch-like style.
- **Efficient**: TorchOpt provides (1) CPU/GPU acceleration differentiable optimizer (2) RPC-based distributed training framework (3) Fast Tree Operations, to largely increase the training efficiency for bi-level optimization problems.

Beyond differentiable optimization, TorchOpt can also be regarded as a functional optimizer that enables [JAX-like](https://github.com/google/jax) composable functional optimizer for PyTorch.
With TorchOpt, users can easily conduct neural network optimization in PyTorch with a functional style optimizer, similar to [Optax](https://github.com/deepmind/optax) in JAX.

--------------------------------------------------------------------------------

The README is organized as follows:

- [TorchOpt as Functional Optimizer](#torchopt-as-functional-optimizer)
  - [Optax-Like API](#optax-like-api)
  - [PyTorch-Like API](#pytorch-like-api)
  - [Differentiable](#differentiable)
- [TorchOpt for Differentiable Optimization](#torchopt-for-differentiable-optimization)
  - [Explicit Gradient (EG)](#explicit-gradient-eg)
  - [Implicit Gradient (IG)](#implicit-gradient-ig)
  - [Zero-order Differentiation (ZD)](#zero-order-differentiation-zd)
- [High-Performance and Distributed Training](#high-performance-and-distributed-training)
  - [CPU/GPU accelerated differentiable optimizer](#cpugpu-accelerated-differentiable-optimizer)
  - [Distributed Training](#distributed-training)
  - [OpTree](#optree)
- [Visualization](#visualization)
- [Examples](#examples)
- [Installation](#installation)
- [Changelog](#changelog)
- [Citing TorchOpt](#citing-torchopt)
- [The Team](#the-team)
- [License](#license)

--------------------------------------------------------------------------------

## TorchOpt as Functional Optimizer

The design of TorchOpt follows the philosophy of functional programming.
Aligned with [`functorch`](https://github.com/pytorch/functorch), users can conduct functional style programming with models, optimizers and training in PyTorch.
We use the Adam optimizer as an example in the following illustration.
You can also check out the tutorial notebook [Functional Optimizer](tutorials/1_Functional_Optimizer.ipynb) for more details.

### Optax-Like API

For those users who prefer fully functional programming, we offer Optax-Like API by passing gradients and optimizer states to the optimizer function.
Here is an example coupled with `functorch`:

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

We also design a base class `torchopt.Optimizer` that has the same interface as `torch.optim.Optimizer`.
We offer origin PyTorch APIs (e.g. `zero_grad()` or `step()`) by wrapping our Optax-Like API for traditional PyTorch users.

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

On top of the same optimization function as `torch.optim`, an important benefit of the functional optimizer is that one can implement differentiable optimization easily.
This is particularly helpful when the algorithm requires differentiation through optimization updates (such as meta-learning practices).
We take as the inputs the gradients and optimizer states, and use non-in-place operators to compute and output the updates.
The processes can be automatically implemented, with the only need from users being to pass the argument `inplace=False` to the functions.
Check out the section [Explicit Gradient (EG)](#explicit-gradient-eg) functional API for example.

--------------------------------------------------------------------------------

## TorchOpt for Differentiable Optimization

We design a bilevel-optimization updating scheme, which can be easily extended to realize various differentiable optimization processes.

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/diffmode.png" width="90%" />
</div>

As shown above, the scheme contains an outer level that has parameters $\phi$ that can be learned end-to-end through the inner level parameters solution $\theta^{\prime}(\phi)$ by using the best-response derivatives $\partial \theta^{\prime}(\phi) / \partial \phi$.
TorchOpt supports three differentiation modes.
It can be seen that the key component of this algorithm is to calculate the best-response (BR) Jacobian.
From the BR-based perspective, existing gradient methods can be categorized into three groups: explicit gradient over unrolled optimization, implicit differentiation, and zero-order gradient differentiation.

### Explicit Gradient (EG)

The idea of the explicit gradient is to treat the gradient step as a differentiable function and try to backpropagate through the unrolled optimization path.
This differentiation mode is suitable for algorithms when the inner-level optimization solution is obtained by a few gradient steps, such as [MAML](https://arxiv.org/abs/1703.03400) and [MGRL](https://arxiv.org/abs/1805.09801).
TorchOpt offers both functional and object-oriented API for EG to fit different user applications.

#### Functional API  <!-- omit in toc -->

The functional API is to conduct optimization in a functional programming style.
Note that we pass the argument `inplace=False` to the functions to make the optimization differentiable.
Refer to the tutorial notebook [Functional Optimizer](tutorials/1_Functional_Optimizer.ipynb) for more guidance.

```python
# Define functional optimizer
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

loss = outer_loss(fmodel, params, meta_params)
meta_grads = torch.autograd.grad(loss, meta_params)
```

#### OOP API  <!-- omit in toc -->

TorchOpt also provides OOP API compatible with the PyTorch programming style.
Refer to the example and the tutorial notebook [Meta-Optimizer](tutorials/3_Meta_Optimizer.ipynb), [Stop Gradient](tutorials/4_Stop_Gradient.ipynb) for more guidance.

```python
# Define meta and inner parameters
meta_params = ...
model = ...
# Define differentiable optimizer
optimizer = torchopt.MetaAdam(model)  # a model instance as the argument instead of model.parameters()

for iter in range(iter_times):
    # Perform inner update
    loss = inner_loss(model, meta_params)
    optimizer.step(loss)

loss = outer_loss(model, meta_params)
loss.backward()
```

### Implicit Gradient (IG)

By treating the solution $\theta^{\prime}$ as an implicit function of $\phi$, the idea of IG is to directly get analytical best-response derivatives $\partial \theta^{\prime} (\phi) / \partial \phi$ by [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem).
This is suitable for algorithms when the inner-level optimal solution is achieved ${\left. \frac{\partial F (\theta, \phi)}{\partial \theta} \right\rvert}_{\theta=\theta^{\prime}} = 0$ or reaches some stationary conditions $F (\theta^{\prime}, \phi) = 0$, such as [iMAML](https://arxiv.org/abs/1909.04630) and [DEQ](https://arxiv.org/abs/1909.01377).
TorchOpt offers both functional and OOP APIs for supporting both [conjugate gradient-based](https://arxiv.org/abs/1909.04630) and [Neumann series-based](https://arxiv.org/abs/1911.02590) IG methods.
Refer to the example [iMAML](https://github.com/waterhorse1/torchopt/tree/readme/examples/iMAML) and the notebook [Implicit Gradient](tutorials/5_Implicit_Differentiation.ipynb) for more guidance.

#### Functional API  <!-- omit in toc -->

For the implicit gradient, similar to [JAXopt](https://jaxopt.github.io/stable/implicit_diff.html), users need to define the stationary condition and TorchOpt provides the decorator to wrap the solve function for enabling implicit gradient computation.

```python
# The stationary condition for the inner-loop
def stationary(params, meta_params, data):
    # Stationary condition construction
    return stationary condition

# Decorator for wrapping the function
# Optionally specify the linear solver (conjugate gradient or Neumann series)
@torchopt.diff.implicit.custom_root(stationary, solve=linear_solver)
def solve(params, meta_params, data):
    # Forward optimization process for params
    return output

# Define params, meta_params and get data
params, meta_prams, data = ..., ..., ...
optimal_params = solve(params, meta_params, data)
loss = outer_loss(optimal_params)

meta_grads = torch.autograd.grad(loss, meta_params)
```

#### OOP API  <!-- omit in toc -->

TorchOpt also offers an OOP API, which users need to inherit from the class `torchopt.nn.ImplicitMetaGradientModule` to construct the inner-loop network.
Users need to define the stationary condition/objective function and the inner-loop solve function to enable implicit gradient computation.

```python
# Inherited from the class ImplicitMetaGradientModule
# Optionally specify the linear solver (conjugate gradient or Neumann series)
class InnerNet(ImplicitMetaGradientModule, linear_solve=linear_solver):
    def __init__(self, meta_param):
        super().__init__()
        self.meta_param = meta_param
        ...

    def forward(self, batch):
        # Forward process
        ...

    def optimality(self, batch, labels):
        # Stationary condition construction for calculating implicit gradient
        # NOTE: If this method is not implemented, it will be automatically
        # derived from the gradient of the `objective` function.
        ...

    def objective(self, batch, labels):
        # Define the inner-loop optimization objective
        ...

    def solve(self, batch, labels):
        # Conduct the inner-loop optimization
        ...

# Get meta_params and data
meta_params, data = ..., ...
inner_net = InnerNet(meta_params)

# Solve for inner-loop process related to the meta-parameters
optimal_inner_net = inner_net.solve(data)

# Get outer loss and solve for meta-gradient
loss = outer_loss(optimal_inner_net)
meta_grads = torch.autograd.grad(loss, meta_params)
```

### Zero-order Differentiation (ZD)

When the inner-loop process is non-differentiable or one wants to eliminate the heavy computation burdens in the previous two modes (brought by Hessian), one can choose Zero-order Differentiation (ZD).
ZD typically gets gradients based on zero-order estimation, such as finite-difference, or [Evolutionary Strategy](https://arxiv.org/abs/1703.03864).
Instead of optimizing the objective $F$, ES optimizes a smoothed objective.
TorchOpt provides both functional and OOP APIs for the ES method.
Refer to the tutorial notebook [Zero-order Differentiation](tutorials/6_Zero_Order_Differentiation.ipynb) for more guidance.

#### Functional API  <!-- omit in toc -->

For zero-order differentiation, users need to define the forward pass calculation and the noise sampling procedure. TorchOpt provides the decorator to wrap the forward function for enabling zero-order differentiation.

```python
# Customize the noise sampling function in ES
def distribution(sample_shape):
    # Generate a batch of noise samples
    # NOTE: The distribution should be spherical symmetric and with a constant variance of 1.
    ...
    return noise_batch

# Distribution can also be an instance of `torch.distributions.Distribution`, e.g., `torch.distributions.Normal(...)`
distribution = torch.distributions.Normal(loc=0, scale=1)

# Specify method and hyper-parameter of ES
@torchopt.diff.zero_order(distribution, method)
def forward(params, batch, labels):
    # Forward process
    ...
    return objective  # the returned tensor should be a scalar tensor
```

#### OOP API  <!-- omit in toc -->

TorchOpt also offers an OOP API, which users need to inherit from the class `torchopt.nn.ZeroOrderGradientModule` to construct the network as an `nn.Module` following a classical PyTorch style.
Users need to define the forward process zero-order gradient procedures `forward()` and a noise sampling function `sample()`.

```python
# Inherited from the class ZeroOrderGradientModule
# Optionally specify the `method` and/or `num_samples` and/or `sigma` used for sampling
class Net(ZeroOrderGradientModule, method=method, num_samples=num_samples, sigma=sigma):
    def __init__(self, ...):
        ...

    def forward(self, batch):
        # Forward process
        ...
        return objective  # the returned tensor should be a scalar tensor

    def sample(self, sample_shape=torch.Size()):
        # Generate a batch of noise samples
        # NOTE: The distribution should be spherical symmetric and with a constant variance of 1.
        ...
        return noise_batch

# Get model and data
net = Net(...)
data = ...

# Forward pass
loss = Net(data)
# Backward pass using zero-order differentiation
grads = torch.autograd.grad(loss, net.parameters())
```

--------------------------------------------------------------------------------

## High-Performance and Distributed Training

### CPU/GPU accelerated differentiable optimizer

We take the optimizer as a whole instead of separating it into several basic operators (e.g., `sqrt` and `div`).
Therefore, by manually writing the forward and backward functions, we can perform the symbolic reduction.
In addition, we can store some intermediate data that can be reused during the backpropagation.
We write the accelerated functions in C++ OpenMP and CUDA, bind them by [`pybind11`](https://github.com/pybind/pybind11) to allow they can be called by Python, and then define the forward and backward behavior using `torch.autograd.Function`.
Users can use it by simply setting the `use_accelerated_op` flag as `True`.
Refer to the corresponding sections in the tutorials [Functional Optimizer](tutorials/1_Functional_Optimizer.ipynb)](tutorials/1_Functional_Optimizer.ipynb) and [Meta-Optimizer](tutorials/3_Meta_Optimizer.ipynb)

```python
optimizer = torchopt.MetaAdam(model, lr, use_accelerated_op=True)
```

### Distributed Training

`TorchOpt` provides distributed training features based on the PyTorch RPC module for better training speed and multi-node multi-GPU support.
Different from the MPI-like parallelization paradigm, which uses multiple homogeneous workers and requires carefully designed communication hooks, the RPC APIs allow users to build their optimization pipeline more flexibly.
Experimental results show that we achieve an approximately linear relationship between the speed-up ratio and the number of workers.
Check out the [Distributed Training Documentation](https://torchopt.readthedocs.io/en/latest/distributed/distributed.html) and [distributed MAML example](https://github.com/metaopt/torchopt/tree/main/examples/distributed/few-shot) for more specific guidance.

### OpTree

We implement the *PyTree* to enable fast nested structure flattening using C++.
The tree operations (e.g., flatten and unflatten) are very important in enabling functional and Just-In-Time (JIT) features of deep learning frameworks.
By implementing it in C++, we can use some cache/memory-friendly structures (e.g., `absl::InlinedVector`) to improve the performance.
For more guidance and comparison results, please refer to our open-source project [`OpTree`](https://github.com/metaopt/optree).

--------------------------------------------------------------------------------

## Visualization

Complex gradient flow in meta-learning brings in a great challenge for managing the gradient flow and verifying its correctness of it.
TorchOpt provides a visualization tool that draws variable (e.g., network parameters or meta-parameters) names on the gradient graph for better analysis.
The visualization tool is modified from [`torchviz`](https://github.com/szagoruyko/pytorchviz).
Refer to the example [visualization code](examples/visualize.py) and the tutorial notebook [Visualization](tutorials/2_Visualization.ipynb) for more details.

The figure below shows the visualization result.
Compared with [`torchviz`](https://github.com/szagoruyko/pytorchviz), TorchOpt fuses the operations within the `Adam` together (orange) to reduce the complexity and provide simpler visualization.

<div align="center">
  <img src="https://github.com/metaopt/torchopt/raw/HEAD/image/torchviz-vs-torchopt.jpg" width="80%" />
</div>

--------------------------------------------------------------------------------

## Examples

In the [`examples`](examples) directory, we offer several examples of functional optimizers and lightweight meta-learning examples with TorchOpt.

- [Model-Agnostic Meta-Learning (MAML) - Supervised Learning](https://arxiv.org/abs/1703.03400) (ICML 2017)
- [Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/abs/1803.09050) (ICML 2018)
- [Model-Agnostic Meta-Learning (MAML) - Reinforcement Learning](https://arxiv.org/abs/1703.03400) (ICML 2017)
- [Meta-Gradient Reinforcement Learning (MGRL)](https://arxiv.org/abs/1805.09801) (NeurIPS 2018)
- [Learning through opponent learning process (LOLA)](https://arxiv.org/abs/1709.04326) (AAMAS 2018)
- [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630) (NeurIPS 2019)

Also, check [`examples`](examples) for more distributed/visualization/functorch-compatible examples.

--------------------------------------------------------------------------------

## Installation

Requirements

- PyTorch
- (Optional) For visualizing computation graphs
  - [Graphviz](https://graphviz.org/download) (for Linux users use `apt/yum install graphviz` or `conda install -c anaconda python-graphviz`)

**Please follow the instructions at <https://pytorch.org> to install PyTorch in your Python environment first.**
Then run the following command to install TorchOpt from PyPI ([![PyPI](https://img.shields.io/pypi/v/torchopt?label=pypi&logo=pypi)](https://pypi.org/project/torchopt) / ![Status](https://img.shields.io/pypi/status/torchopt?label=status)):

```bash
pip3 install torchopt
```

If the minimum version of PyTorch is not satisfied, `pip` will install/upgrade it for you. Please be careful about the `torch` build for CPU / CUDA support (e.g. `cpu`, `cu116`, `cu117`).
You may need to specify the extra index URL for the `torch` package:

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

We provide a [conda](https://github.com/conda/conda) environment recipe to install the build toolchain such as `cmake`, `g++`, and `nvcc`.
You can use the following commands with [`conda`](https://github.com/conda/conda) / [`mamba`](https://github.com/mamba-org/mamba) to create a new isolated environment.

```bash
git clone https://github.com/metaopt/torchopt.git
cd torchopt

# You may need `CONDA_OVERRIDE_CUDA` if conda fails to detect the NVIDIA driver (e.g. in docker or WSL2)
CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda-recipe-minimal.yaml

conda activate torchopt
make install-editable  # or run `pip3 install --no-build-isolation --editable .`
```

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

--------------------------------------------------------------------------------

## Citing TorchOpt

If you find TorchOpt useful, please cite it in your publications.

```bibtex
@article{torchopt,
  title   = {TorchOpt: An Efficient Library for Differentiable Optimization},
  author  = {Ren, Jie and Feng, Xidong and Liu, Bo and Pan, Xuehai and Fu, Yao and Mai, Luo and Yang, Yaodong},
  journal = {arXiv preprint arXiv:2211.06934},
  year    = {2022}
}
```

## The Team

TorchOpt is a work by [Jie Ren](https://github.com/JieRen98), [Xidong Feng](https://github.com/waterhorse1), [Bo Liu](https://benjamin-eecs.github.io/), [Xuehai Pan](https://github.com/XuehaiPan), [Luo Mai](https://luomai.github.io), and [Yaodong Yang](https://www.yangyaodong.com).

## License

TorchOpt is released under the Apache License, Version 2.0.
