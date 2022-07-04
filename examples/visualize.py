import torch
import torchviz
from torch import nn
from torch.nn import functional as F

import TorchOpt


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x, meta_param):
        return self.fc(x) + meta_param


def draw_torchviz():
    net = Net(dim).cuda()
    optimizer = TorchOpt.MetaAdam(net, lr=1e-3, use_accelerated_op=False)
    meta_param = torch.tensor(1., requires_grad=True)

    xs = torch.ones(batch_size, dim).cuda()

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    optimizer.step(loss)

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # draw computation graph
    torchviz.make_dot(loss).render("torchviz_graph", format="svg")


def draw_TorchOpt():
    net = Net(dim).cuda()
    optimizer = TorchOpt.MetaAdam(net, lr=1e-3, use_accelerated_op=True)
    meta_param = torch.tensor(1., requires_grad=True)

    xs = torch.ones(batch_size, dim).cuda()

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # set enable_visual
    net_state_0 = TorchOpt.extract_state_dict(net,
                                              enable_visual=True,
                                              visual_prefix='step0.')
    optimizer.step(loss)
    # set enable_visual
    net_state_1 = TorchOpt.extract_state_dict(net,
                                              enable_visual=True,
                                              visual_prefix='step1.')

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # draw computation graph
    TorchOpt.visual.make_dot(
        loss, [net_state_0, net_state_1, {
            meta_param: "meta_param"
        }]).render("TorchOpt_graph", format="svg")


if __name__ == '__main__':
    dim = 5
    batch_size = 2
    draw_torchviz()
    draw_TorchOpt()
