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

import functorch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchopt
from helpers.omniglot_loaders import OmniglotNShot
from torchopt import pytree


mpl.use('Agg')
plt.style.use('bmh')


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
    fnet, meta_params = model = functorch.make_functional(net)
    meta_opt = torchopt.adam(lr=1e-3)
    meta_opt_state = meta_opt.init(meta_params)

    log = []
    test(db, model, epoch=-1, log=log, args=args)
    for epoch in range(10):
        meta_opt, meta_opt_state = train(db, model, (meta_opt, meta_opt_state), epoch, log, args)
        test(db, model, epoch, log, args)
        plot(log)


def train(db, model, meta_opt_and_state, epoch, log, args):
    n_train_iter = db.x_train.shape[0] // db.batchsz
    fnet, meta_params = model
    meta_opt, meta_opt_state = meta_opt_and_state
    # Given this module we've created, rip out the parameters and buffers
    # and return a functional version of the module. `fnet` is stateless
    # and can be called with `fnet(params, buffers, args, kwargs)`
    # fnet, params, buffers = functorch.make_functional_with_buffers(net)

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num = x_spt.size(0)

        n_inner_iter = args.inner_steps
        reg_param = args.reg_params

        qry_losses = []
        qry_accs = []

        for i in range(task_num):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.

            init_params = pytree.tree_map(
                lambda t: t.clone().detach_().requires_grad_(requires_grad=t.requires_grad),
                meta_params,
            )
            optimal_params = train_imaml_inner_solver(
                init_params,
                meta_params,
                (x_spt[i], y_spt[i]),
                (fnet, n_inner_iter, reg_param),
            )
            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = fnet(optimal_params, x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).float().mean()
            qry_losses.append(qry_loss)
            qry_accs.append(qry_acc.item())

        qry_losses = torch.mean(torch.stack(qry_losses))
        meta_grads = torch.autograd.grad(qry_losses, meta_params)
        meta_updates, meta_opt_state = meta_opt.update(meta_grads, meta_opt_state)
        meta_params = torchopt.apply_updates(meta_params, meta_updates)
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

    return (meta_opt, meta_opt_state)


def test(db, model, epoch, log, args):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    fnet, meta_params = model
    n_test_iter = db.x_test.shape[0] // db.batchsz

    n_inner_iter = args.inner_steps
    reg_param = args.reg_params
    qry_losses = []
    qry_accs = []

    for _ in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')

        task_num = x_spt.size(0)

        for i in range(task_num):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.

            init_params = pytree.tree_map(
                lambda t: t.clone().detach_().requires_grad_(requires_grad=t.requires_grad),
                meta_params,
            )
            optimal_params = test_imaml_inner_solver(
                init_params,
                meta_params,
                (x_spt[i], y_spt[i]),
                (fnet, n_inner_iter, reg_param),
            )

            # The query loss and acc induced by these parameters.
            qry_logits = fnet(optimal_params, x_qry[i])
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


def imaml_objective(params, meta_params, data, aux):
    x_spt, y_spt = data
    fnet, n_inner_iter, reg_param = aux
    y_pred = fnet(params, x_spt)
    regularization_loss = 0
    for p1, p2 in zip(params, meta_params):
        regularization_loss += 0.5 * reg_param * torch.sum(torch.square(p1 - p2))
    loss = F.cross_entropy(y_pred, y_spt) + regularization_loss
    return loss


@torchopt.diff.implicit.custom_root(
    functorch.grad(imaml_objective, argnums=0),
    argnums=1,
    has_aux=False,
    solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
)
def train_imaml_inner_solver(params, meta_params, data, aux):
    x_spt, y_spt = data
    fnet, n_inner_iter, reg_param = aux
    # Initial functional optimizer based on TorchOpt
    inner_opt = torchopt.sgd(lr=1e-1)
    inner_opt_state = inner_opt.init(params)
    with torch.enable_grad():
        # Temporarily enable gradient computation for conducting the optimization
        for _ in range(n_inner_iter):
            pred = fnet(params, x_spt)
            loss = F.cross_entropy(pred, y_spt)  # compute loss
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params, meta_params):
                regularization_loss += 0.5 * reg_param * torch.sum(torch.square(p1 - p2))
            final_loss = loss + regularization_loss
            grads = torch.autograd.grad(final_loss, params)  # compute gradients
            updates, inner_opt_state = inner_opt.update(
                grads,
                inner_opt_state,
                inplace=True,
            )  # get updates
            params = torchopt.apply_updates(params, updates, inplace=True)
    return params


def test_imaml_inner_solver(params, meta_params, data, aux):
    x_spt, y_spt = data
    fnet, n_inner_iter, reg_param = aux
    # Initial functional optimizer based on TorchOpt
    inner_opt = torchopt.sgd(lr=1e-1)
    inner_opt_state = inner_opt.init(params)
    with torch.enable_grad():
        # Temporarily enable gradient computation for conducting the optimization
        for _ in range(n_inner_iter):
            pred = fnet(params, x_spt)
            loss = F.cross_entropy(pred, y_spt)  # compute loss
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params, meta_params):
                regularization_loss += 0.5 * reg_param * torch.sum(torch.square(p1 - p2))
            final_loss = loss + regularization_loss
            grads = torch.autograd.grad(final_loss, params)  # compute gradients
            updates, inner_opt_state = inner_opt.update(
                grads,
                inner_opt_state,
                inplace=True,
            )  # get updates
            params = torchopt.apply_updates(params, updates, inplace=True)
    return params


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
    ax.set_title('iMAML Omniglot (Functional)')
    ax.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'imaml-accs-functional.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()
