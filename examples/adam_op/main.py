from typing import Any

import torch
import jax
from torch import nn

import OpTorch
from OpTorch import adam as old_adam
import functorch
from torchviz import make_dot
from OpTorch._src.custom_op import AdamOp as MyAdam


DIM = 1024


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM, bias=True)
        self.fc2 = nn.Linear(DIM, 1, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    BATCH = 10
    lr = 1e-3

    data = torch.rand(BATCH, DIM, device=device)
    net = Net()
    net = net.to(device)
    func, Tparams = functorch.make_functional(net)

    old_adam_inst = old_adam(lr=lr, requires_grad=True)
    old_adam_state = old_adam_inst.init(Tparams)

    Tloss = func(Tparams, data).sum()
    Tgrad = torch.autograd.grad(Tloss, Tparams, create_graph=True)
    Tupdates, old_adam_state = old_adam_inst.update(
        Tgrad, old_adam_state, inplace=False)
    Tnew_params = OpTorch.apply_updates(Tparams, Tupdates, inplace=False)
    Tloss = func(Tnew_params, data).sum()
    for i in range(1):
        Tloss.backward(retain_graph=True)
    make_dot(Tloss).render("origin", format="pdf")

    torch.manual_seed(0)
    data = torch.rand(BATCH, DIM, device=device)
    net = Net()
    net = net.to(device)

    func, params = functorch.make_functional(net)
    mus = jax.tree_map(  # First moment
        lambda t: torch.zeros_like(t, requires_grad=True), params)
    nus = jax.tree_map(  # Second moment
        lambda t: torch.zeros_like(t, requires_grad=True), params)
    adamOp = MyAdam(1, lr=lr)

    loss = func(params, data).sum()
    updates = torch.autograd.grad(loss, params, create_graph=True)

    out = jax.tree_map(adamOp, updates, mus, nus)
    new_updates = []
    new_mu = []
    new_nu = []
    for update, (mu, nu) in out:
        new_updates.append(update)
        new_mu.append(mu)
        new_nu.append(nu)
    new_params = OpTorch.apply_updates(
        params, tuple(new_updates), inplace=False)
    loss = func(new_params, data).sum()
    for i in range(1):
        loss.backward(retain_graph=True)
    make_dot(loss).render("custom", format="pdf")

    print(f"torch result: {Tparams[0].grad}")
    print(f"custom op result: {params[0].grad}")


if __name__ == '__main__':
    main()
