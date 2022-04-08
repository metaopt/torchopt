# This file is modified from:
# https://github.com/alexis-jacq/LOLA_DiCE

import torch
import torch.nn as nn
import TorchOpt


class theta_model(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(
            theta.detach(), requires_grad=True))


class Agent():
    def __init__(self, args):

        self.args = args
        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=args.lr_out)

        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,), lr=args.lr_v)

        self.set_virtual()

    def set_virtual(self):
        self.virtual_theta = theta_model(self.theta)
        self.virtual_optimiser = TorchOpt.MetaSGD(
            self.virtual_theta, lr=self.args.lr_in)

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
