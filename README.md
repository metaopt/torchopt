
<div align="center">
<img src=image/logod-07.png width=75% />
</div>

**TorchOpt** is a high-performance optimizer library built upon [PyTorch](https://pytorch.org/) for easy implementation of functional optimization and gradient-based meta-learning. It consists of two main features:
- TorchOpt provides functional optimizer which enables [JAX-like](https://github.com/google/jax) composable functional optimizer for PyTorch. With TorchOpt, one can easily conduct neural network optimization in PyTorch with functional style optimizer, similar to  [Optax](https://github.com/deepmind/optax) in JAX.
- With the desgin of functional programing, TorchOpt provides efficient, flexible, and easy-to-implement differentiable optimizer for gradient-based meta-learning research. It largely reduces the efforts required to implement sophisticated meta-learning algorithms.

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
- [The Team](#the-team)
- [Citing TorchOpt](#citing-torchopt)


## TorchOpt as Functional Optimizer
The desgin of TorchOpt follows the philosophy of functional programming. Aligned with [functorch](https://github.com/pytorch/functorch), users can conduct functional style programing with models, optimizers and training in PyTorch. We use the Adam optimizer as an example in the following illustration. You can also check out the tutorial notebook [Functional Optimizer](./tutorials/1_Functional_Optimizer.ipynb) for more details.
### Optax-Like API
For those users who prefer fully functional programing, we offer Optax-Like API by passing gradients and optimizers states to the optimizer function. We design base class `torchopt.Optimizer` that has the same interface as `torch.optim.Optimizer`. Here is an example coupled with functorch:
```python
import torch
from torch import nn
from torch import data
from nn import functional as F
import functorch
import torchopt

class Net(nn.Module):...

class Loader(data.DataLoader):...

net = Net() # init
loader = Loader()
optimizer = torchopt.adam()
func, params = functorch.make_functional(net)  # use functorch extract network parameters
opt_state = optimizer.init(params)  # init optimizer
xs, ys = next(loader)  # get data
pred = func(params, xs)  # forward
loss = F.cross_entropy(pred, ys)  # compute loss
grad = torch.autograd.grad(loss, params)  # compute gradients
updates, opt_state = optimizer.update(grad, opt_state)  # get updates
params = torchopt.apply_updates(params, updates)  # update network parameters
```
### PyTorch-Like API
We also offer origin PyTorch APIs (e.g. `zero_grad()` or `step()`) by warpping our Optax-Like API for traditional PyTorch user:
<!-- The functional programming can easily disguise as origin PyTorch APIs (e.g. `zero_grad()` or `step()`), the only we need is to build a new class that contains both the optimizer function and optimizer states. -->
```python
net = Net()  # init
loader = Loader()
optimizer = torchopt.Adam(net.parameters())
xs, ys = next(loader)  # get data
pred = net(xs)  # forward
loss = F.cross_entropy(pred, ys)  # compute loss
optimizer.zero_grad()  # zero gradients
loss.backward()  # backward
optimizer.step()  # step updates
```
### Differentiable
On top of the same optimization function as `torch.optim`, an important benefit of functional optimizer is that one can implement differentiable optimization easily. This is particularly helpful when the algorithm requires to differentiate through optimization update (such as meta learning practices). We take as the inputs the gradients and optimizer states, use non-in-place operators to compute and output the updates. The processes can be automatically implemented, with the only need from users being to pass the argument `inplace=False` to the functions:
```python
# get updates
updates, opt_state = optimizer.update(grad, opt_state, inplace=False)
# update network parameters
params = torchopt.apply_updates(params, updates, inplace=False)
```
## TorchOpt as Differentiable Optimizer for Meta-Learning
Meta-Learning has gained enormous attention in both Supervised Learning and Reinforcement Learning. Meta-Learning algorithms often contain a bi-level optimisation process with *inner loop* updating the network parameters and *outer loop* updating meta parameters. The figure below illustrates the basic formulation for meta-optimization in Meta-Learning. The main feature is that the gradients of *outer loss* will back-propagate through all `inner.step` operations.
<div align="center">
<img src=/image/TorchOpt.png width=85% />
</div>

Since network parameters become a node of computation graph, a flexible Meta-Learning library should enable users manually control the gradient graph connection which means that users should have access to the network parameters and optimizer states for manually detaching or connecting the computation graph. In PyTorch designing, the network parameters or optimizer states are members of network (a.k.a. `nn.Module`) or optimizer (a.k.a. `optim.Optimizer`), this design significantly introducing difficulty for user control network parameters or optimizer states. Previous differentiable optimizer Repo [higher](https://github.com/facebookresearch/higher), [learn2learn](https://github.com/learnables/learn2learn) follows the PyTorch designing which leads to inflexible API.

In contrast to them, TorchOpt realizes differentiable optimizer with functional programing, where Meta-Learning researchers could control the network parameters or optimizer states as normal variables (a.k.a. `torch.Tensor`). This functional optimizer design of TorchOpt is beneficial for implementing complex gradient flow Meta-Learning algorithms and allow us to improve computational efficiency by using techniques like operator fusion.

<!-- The biggest difference of implementing Meta-Learning algorithms between others is that the network parameters are not [leaf variables](https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html?highlight=leaf#torch.Tensor.is_leaf) for backpropagating the *outer loss*. This difference requires a differentiable optimizer that updates the network parameters using a non-[in-place](https://discuss.pytorch.org/t/what-is-in-place-operation/16244) manner for preserving *inner loop*'s computation graph.

Since network parameters become a node of computation graph, a flexible meta-learning library should enable users manually control the gradient graph connection which means that users should have access to the network parameters and optimizer states for manually detaching or connecting the computation graph. In the PyTorch design, the network parameters or optimizer states are members of network (a.k.a. `nn.Module`) or optimizer (a.k.a. `optim.Optimizer`), this design incurs difficulties for user to control network parameters or optimizer states.

We hope meta-learning researchers could control the network parameters or optimizer states as normal variables (a.k.a. `torch.Tensor`). Inspired by [Optax](https://github.com/deepmind/optax), we think designing a functional style optimizer that treat network parameters or optimizer states as variables instead of class members, which mathces our demond of making network parameters or optimizer states. This design would be beneficial for implementing complex gradient flow meta-learning algorithms and allow us to dig potential performance by using techniques like operator fusion. -->

### Meta-Learning API
<!-- Meta-Learning algorithms often use *inner loop* to update network parameters and compute an *outer loss* then back-propagate the *outer loss*. So the optimizer used in the *inner loop* should be differentiable. Thanks to the functional design, we can easily realize this requirement. -->
- We design a base class `torchopt.MetaOptimizer` for managing network updates in Meta-Learning. The constructor of `MetaOptimizer` takes as input the network rather than network parameters. `MetaOptimizer` exposed interface `step(loss)` takes as input the loss for step the network parameter. Refer to the tutorial notebook [Meta Optimizer](./tutorials/2_Meta_Optimizer.ipynb) for more details.
- We offer `torchopt.chain` which can apply a list of chainable update transformations. Combined with `MetaOptimizer`, it can help you conduct gradient transformation such as gradient clip before the Meta optimizer steps. Refer to the tutorial notebook [Meta Optimizer](./tutorials/2_Meta_Optimizer.ipynb) for more details.
- We observe that different Meta-Learning algorithms vary in inner-loop parameter recovery. TorchOpt provides basic functions for users to extract or recover network parameters and optimizer states anytime anywhere they want.
- Some algorithms such as [MGRL](https://proceedings.neurips.cc/paper/2018/file/2715518c875999308842e3455eda2fe3-Paper.pdf) initialize the inner-loop parameters inherited from previous inner-loop process when conducting a new bi-level process. TorchOpt also provides a finer function `stop_gradient` for manipulating the gradient graph, which is helpful for this kind of algortihms. Refer to the notebook [Stop Gradient](./tutorials/4_Stop_Gradient.ipynb) for more details.

We give an example of [MAML](https://arxiv.org/abs/1703.03400) with inner-loop Adam optimizer to illustrate TorchOpt APIs:

```python
net = Net() # init
# the constructor `MetaOptimizer` takes as input the network
inner_optim = torchopt.MetaAdam(net)
outer_optim = torchopt.Adam(net.parameters())

for train_iter in range(train_iters):
    outer_loss = 0
    for task in range(tasks):
        loader = Loader(tasks)

        # store states at the inital points
        net_state = torchopt.extract_state_dict(net) # extract state
        optim_state = torchopt.extract_state_dict(inner_optim)
        for inner_iter in range(inner_iters):
            # compute inner loss and perform inner update
            xs, ys = next(loader)
            pred = net(xs)
            inner_loss = F.cross_entropy(pred, ys)
            inner_optim.step(inner_loss)
        # compute outer loss and back-propagate
        xs, ys = next(loader)
        pred = net(xs)
        outer_loss += F.cross_entropy(pred, ys)

        # recover network and optimizer states at the inital point for the next task
        torchopt.recover_state_dict(inner_optim, optim_state)
        torchopt.recover_state_dict(net, net_state)

    outer_loss /= len(tasks) # task average
    outer_optim.zero_grad()
    outer_loss.backward()
    outer_optim.step()

    # stop gradient if necessary
    torchopt.stop_gradient(net)
    torchopt.stop_gradient(inner_optim)
```
## Examples
In *examples/*, we offer serveral examples of functional optimizer and 5 light-weight meta-learning examples with TorchOpt. The meta-learning examples covers 2 Supervised Learning and 3 Reinforcement Learning algorithms.
- [Model Agnostic Meta Learning (MAML)-Supervised Learning](https://arxiv.org/abs/1703.03400) (ICML2017)
- [Learning to Reweight Examples for Robust Deep Learning](https://arxiv.org/pdf/1803.09050.pdf) (ICML2018)
- [Model Agnostic Meta Learning (MAML)-Reinforcement Learning](https://arxiv.org/abs/1703.03400) (ICML2017)
- [Meta Gradient Reinforcement Learning (MGRL)](https://proceedings.neurips.cc/paper/2018/file/2715518c875999308842e3455eda2fe3-Paper.pdf) (NeurIPS 2018)
- [Learning through opponent learning process (LOLA)](https://arxiv.org/abs/1709.04326) (AAMAS 2018)

## High-Performance
One can think of the scale procedures on gradients of optimizer algorithms as a combination of several operations. For example, the implementation of the Adam algorithm often includes addition, multiplication, power and square operations, one can fuse these operations into several compound functions. The operator fusion could greatly simplify the computation graph and reduce the GPU function launching stall. In addition, one can also implement the optimizer backward function and manually reuse some intermediate tensors to improve the backward performance. Users can pass argument `use_accelerated_op=True` to `adam`, `Adam` and `MetaAdam` to enable the fused accelerated operator. The arguments are the same between the two kinds of implementations.

Here we evaluate the performance using the maml-omniglot code with the inner-loop Adam optimizer on GPU. We comparble the run time of the overall algorithm and the meta-optimization (outer-loop optimization) under different network architecture/inner-step numbers. We choose [higher](https://github.com/facebookresearch/higher) as our baseline. The figure below illustrate that our accelerated Adam can achieve at least 1/3 efficiency improvement over the baseline.
<div align="center">
<img src=image/time.png width=80% />
</div>

Notably, the operator fusion not only increases performance but also help simplify the computation graph, which will be discussed in the next section.

## Visualization
Complex gradient flow in meta-learning brings in a great challenge for managing the gradient flow and verifying the correctness of it. TorchOpt provides a visualization tool that draw variable (e.g. network parameters or meta parameters) names on the gradient graph for better analyzing. The visualization tool is modified from [torchviz](https://github.com/szagoruyko/pytorchviz). We provide an example using the [visualization code](./examples/visualize.py). Also refer to the notebook [Visualization](./tutorials/3_Visualization.ipynb) for more details.

The figure below show the visulization result. Compared with torchviz, TorchOpt fuses the operations within the Adam together (orange) to reduce the complexity and provide simpler visualization.

<div align="center">
<img src=image/torchviz_torchopt.jpg width=80% />
</div>

## Installation
Requirements
  - (Optional) For visualizing computation graphs
    - [Graphviz](https://graphviz.org/download/) (for Linux users use `apt/yum install graphviz` or `conda install -c anaconda python-graphviz`)
```bash
pip install torchopt
```

You can also build shared libraries from source, use:
```bash
git clone git@github.com:metaopt/torchopt.git
cd torchopt
python setup.py build_from_source
```
## Future Plan
- [ ] Support general implicit differentiation with functional programing.
- [ ] Support more optimizers such as AdamW, RMSPROP
- [ ] CPU-acclerated optimizer

## The Team
TorchOpt is a work by Jie Ren, Xidong Feng, [Bo Liu](https://github.com/Benjamin-eecs/), [Luo Mai](https://luomai.github.io/) and [Yaodong Yang](https://www.yangyaodong.com/).

## Citing TorchOpt

If you find TorchOpt useful, please cite it in your publications.

```
@software{TorchOpt,
  author = {Jie Ren and Xidong Feng and Bo Liu and Luo Mai and Yaodong Yang},
  title = {TorchOpt},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/metaopt/torchopt}},
}
```
