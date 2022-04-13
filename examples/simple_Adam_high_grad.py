import torch
import functorch
import torch.autograd
from torch import nn

import OpTorch


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=False)
        self.fc.weight.data = torch.ones_like(self.fc.weight.data)

    def forward(self, x, meta_param):
        return self.fc(x) + meta_param


def main():
    batch_size = 1
    dim = 1
    net = Net(dim)
    func, params = functorch.make_functional(net)

    lr = 1.
    optimizer = OpTorch.sgd(lr, requires_grad=True)
    meta_param = torch.tensor(1., requires_grad=True)

    opt_state = optimizer.init(params)

    xs = torch.ones(batch_size, dim)
    ys = torch.ones(batch_size)

    pred = func(params, xs, meta_param)
    loss = ((pred - ys) ** 2).sum()
    grad = torch.autograd.grad(loss, params, create_graph=True)
    updates, opt_state = optimizer.update(grad, opt_state, inplace=False)
    params = OpTorch.apply_updates(params, updates, inplace=False)

    pred = func(params, xs, meta_param)
    loss = ((pred - ys) ** 2).sum()
    loss.backward()

    print(meta_param.grad)


if __name__ == '__main__':
    main()
