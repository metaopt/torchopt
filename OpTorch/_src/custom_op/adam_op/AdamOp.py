from typing import Any
import torch
from . import adam_op


class AdamOp(object):
    class MuOp(torch.autograd.Function):
        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            pass

        @staticmethod
        def forward(ctx, *args):
            updates, mu, b1 = args
            new_mu = adam_op.forwardMu(updates, mu, b1)
            ctx.save_for_backward(updates, mu)
            ctx.b1 = b1
            return new_mu

        @staticmethod
        def backward(ctx, *args):
            dmu = args[0]
            updates, mu = ctx.saved_tensors
            b1 = ctx.b1
            result = adam_op.backwardMu(dmu, updates, mu, b1)
            return result[0], result[1], None

    class NuOp(torch.autograd.Function):
        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            pass

        @staticmethod
        def forward(ctx, *args):
            updates, nu, b2 = args
            new_nu = adam_op.forwardNu(updates, nu, b2)
            ctx.save_for_backward(updates, nu)
            ctx.b2 = b2
            return new_nu

        @staticmethod
        def backward(ctx, *args):
            dnu = args[0]
            updates, nu = ctx.saved_tensors
            b2 = ctx.b2
            result = adam_op.backwardNu(dnu, updates, nu, b2)
            return result[0], result[1], None

    class UpdatesOp(torch.autograd.Function):
        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            pass

        @staticmethod
        def forward(ctx, *args):
            new_mu, new_nu, (lr, b1, b2, eps, eps_root, count) = args
            new_updates = adam_op.forwardUpdates(
                new_mu, new_nu, -lr, b1, b2, eps, eps_root, count)
            ctx.save_for_backward(new_updates, new_mu, new_nu)
            ctx.others = (lr, b1, b2, eps, eps_root, count)
            return new_updates

        @staticmethod
        def backward(ctx, *args):
            dupdates = args[0]
            updates, new_mu, new_nu = ctx.saved_tensors
            lr, b1, b2, eps, eps_root, count = ctx.others
            result = adam_op.backwardUpdates(
                dupdates, updates, new_mu, new_nu, -lr, b1, b2, count)
            return result[0], result[1], None

    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, eps_root=0., inplace=True):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root
        self.muOp = self.MuOp()
        self.nuOp = self.NuOp()
        self.updatesOp = self.UpdatesOp()
        self.inplace = inplace

    def __call__(self, mu, nu, updates, count):
        if updates is None:
            return mu, nu, None
        if self.inplace:
            # fix: notably here use -lr as input
            new_updates, new_mu, new_nu = adam_op.forward_(updates, mu, nu, -self.lr, self.b1,
                                                           self.b2, self.eps, self.eps_root, count)
        else:
            new_mu = self.muOp.apply(updates, mu, self.b1)
            new_nu = self.nuOp.apply(updates, nu, self.b2)
            new_updates = self.updatesOp.apply(
                new_mu,
                new_nu,
                (self.lr, self.b1, self.b2, self.eps, self.eps_root, count))
        return new_mu, new_nu, new_updates
