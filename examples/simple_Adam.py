import torch
import functorch
import torch.autograd
from torch import nn
import optax
import jax
from jax import numpy as jnp

import OpTorch


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=False)
        self.fc.weight.data = torch.ones_like(self.fc.weight.data)

    def forward(self, x):
        return self.fc(x)


def origin_jax():
    learning_rate = 1.
    batch_size = 1
    dim = 1
    optimizer = optax.adam(learning_rate)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    params = {'w': jnp.ones((dim, 1))}
    opt_state = optimizer.init(params)

    compute_loss = lambda params, x, y: ((jnp.matmul(x, params['w']) - y) ** 2).sum()

    xs = 2 * jnp.ones((batch_size, dim))
    ys = jnp.ones((batch_size, ))
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    print(params)
    params = optax.apply_updates(params, updates)
    print(params)


def interact_with_functorch():
    batch_size = 1
    dim = 1
    net = Net(dim)
    func, params = functorch.make_functional(net)

    lr = 1.
    optimizer = OpTorch.adam(lr)

    opt_state = optimizer.init(params)

    xs = 2 * torch.ones(batch_size, dim)
    ys = torch.ones(batch_size)

    pred = func(params, xs)
    loss = ((pred - ys) ** 2).sum()
    grad = torch.autograd.grad(loss, params)
    updates, opt_state = optimizer.update(grad, opt_state)
    print(params)
    params = OpTorch.apply_updates(params, updates)
    print(params)


def full_disguise():
    batch_size = 1
    dim = 1
    net = Net(dim)

    lr = 1.
    optim = OpTorch.Adam(net.parameters(), lr=lr)

    xs = 2 * torch.ones(batch_size, dim)
    ys = torch.ones(batch_size)

    pred = net(xs)
    loss = ((pred - ys) ** 2).sum()

    print(net.fc.weight)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(net.fc.weight)


def origin_torch():
    batch_size = 1
    dim = 1
    net = Net(dim)

    lr = 1.
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    xs = 2 * torch.ones(batch_size, dim)
    ys = torch.ones(batch_size)

    pred = net(xs)
    loss = ((pred - ys) ** 2).sum()

    print(net.fc.weight)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(net.fc.weight)


if __name__ == '__main__':
    origin_jax()
    interact_with_functorch()
    full_disguise()
    origin_torch()
