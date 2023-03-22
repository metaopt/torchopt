# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This example shows how to use TorchOpt to do iMAML-GD (see [1] for more details)
for few-shot Omniglot classification.

[1] Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
    Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems (pp. 113-124).
    https://arxiv.org/abs/1909.04630
"""

import argparse
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchopt
from helpers.omniglot_loaders import OmniglotNShot
from torchopt.diff.implicit import ImplicitMetaGradientModule


mpl.use('Agg')
plt.style.use('bmh')


class InnerNet(
    ImplicitMetaGradientModule,
    linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
):
    def __init__(self, meta_net, n_inner_iter, reg_param):
        super().__init__()
        self.meta_net = meta_net
        self.net = torchopt.module_clone(meta_net, by='deepcopy', detach_buffers=True)
        self.n_inner_iter = n_inner_iter
        self.reg_param = reg_param
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                p1.data.copy_(p2.data)
                p1.detach_().requires_grad_()

    def forward(self, x):
        return self.net(x)

    def objective(self, x, y):
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        regularization_loss = 0
        for p1, p2 in zip(self.parameters(), self.meta_parameters()):
            regularization_loss += 0.5 * self.reg_param * torch.sum(torch.square(p1 - p2))
        return loss + regularization_loss

    def solve(self, x, y):
        params = tuple(self.parameters())
        inner_optim = torchopt.SGD(params, lr=1e-1)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(self.n_inner_iter):
                loss = self.objective(x, y)
                inner_optim.zero_grad()
                loss.backward(inputs=params)
                inner_optim.step()
        return self


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--inner_steps', type=int, help='number of inner steps', default=5)
    argparser.add_argument(
        '--reg_params',
        type=float,
        help='regularization parameters',
        default=2.0,
    )
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=16,
    )
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Set up the Omniglot loader.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        rng=rng,
        device=device,
    )

    # Create a vanilla PyTorch neural network.
    net = nn.Sequential(
        nn.Conv2d(1, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True, track_running_stats=False),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True, track_running_stats=False),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True, track_running_stats=False),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, args.n_way),
    ).to(device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    net.train()
    meta_opt = torchopt.Adam(net.parameters(), lr=1e-3)

    log = []
    test(db, net, epoch=-1, log=log, args=args)
    for epoch in range(10):
        train(db, net, meta_opt, epoch, log, args)
        test(db, net, epoch, log, args)
        plot(log)


def train(db, net, meta_opt, epoch, log, args):
    n_train_iter = db.x_train.shape[0] // db.batchsz
    n_inner_iter = args.inner_steps
    reg_param = args.reg_params
    task_num = args.task_num

    inner_nets = [InnerNet(net, n_inner_iter, reg_param) for _ in range(task_num)]
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()

        for i in range(task_num):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.

            inner_net = inner_nets[i]
            inner_net.reset_parameters()
            optimal_inner_net = inner_net.solve(x_spt[i], y_spt[i])

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = optimal_inner_net(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).float().mean()
            qry_losses.append(qry_loss)
            qry_accs.append(qry_acc.item())

        qry_losses = torch.mean(torch.stack(qry_losses))
        qry_losses.backward()
        meta_opt.step()
        qry_losses = qry_losses.item()
        qry_accs = 100.0 * np.mean(qry_accs)
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time

        print(
            f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}',
        )
        log.append(
            {
                'epoch': i,
                'loss': qry_losses,
                'acc': qry_accs,
                'mode': 'train',
                'time': time.time(),
            },
        )


def test(db, net, epoch, log, args):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    # TODO: Maybe pull this out into a separate module so it
    # doesn't have to be duplicated between `train` and `test`?
    n_inner_iter = args.inner_steps
    reg_param = args.reg_params

    for _ in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')

        task_num = x_spt.size(0)

        for i in range(task_num):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.

            inner_net = InnerNet(net, n_inner_iter, reg_param)
            with torch.no_grad():
                optimal_inner_net = inner_net.solve(x_spt[i], y_spt[i])

            # The query loss and acc induced by these parameters.
            qry_logits = optimal_inner_net(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).float().mean()
            qry_losses.append(qry_loss.item())
            qry_accs.append(qry_acc.item())

    qry_losses = np.mean(qry_losses)
    qry_accs = 100.0 * np.mean(qry_accs)

    print(f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
    log.append(
        {
            'epoch': epoch + 1,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'test',
            'time': time.time(),
        },
    )


def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(80, 100)
    ax.set_title('iMAML Omniglot')
    ax.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'imaml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()
