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
# This file is modified from:
# https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py
# ==============================================================================
# Copyright (c) Facebook, Inc. and its affiliates.
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
"""
This example shows how to use TorchOpt to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400
This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import argparse
import copy
import os
import random
import threading
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from setproctitle import getproctitle, setproctitle

import torchopt
import torchopt.distributed as todist
from helpers.omniglot_loaders import OmniglotNShot


mpl.use('Agg')
plt.style.use('bmh')


LOCK = threading.Lock()
LOCAL_DATA_LOADER = None
TASK_DATA_LOADERS = {}
LOCAL_DEVICE = None


def worker_init():
    global LOCAL_DEVICE

    world_info = todist.get_world_info()

    proctitle = f'{world_info.worker_name}: {getproctitle().strip()}'
    print(f'Worker init:=> {proctitle}')
    setproctitle(proctitle)

    seed = world_info.world_rank
    local_rank = world_info.local_rank

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if world_info.local_rank < torch.cuda.device_count():
        torch.cuda.set_device(world_info.local_rank)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
    else:
        device = None
    LOCAL_DEVICE = device


def build_model(args, device):
    return nn.Sequential(
        nn.Conv2d(1, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1.0, affine=True),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, args.n_way),
    ).to(device)


def set_local_data_loader(args, device):
    global LOCAL_DATA_LOADER

    if LOCAL_DATA_LOADER is None:
        rng = np.random.default_rng(args.seed)

        with LOCK:
            LOCAL_DATA_LOADER = OmniglotNShot(
                '/tmp/omniglot-data',
                batchsz=args.task_num,
                n_way=args.n_way,
                k_shot=args.k_spt,
                k_query=args.k_qry,
                imgsz=28,
                rng=rng,
                device=device,
            )

    return LOCAL_DATA_LOADER


def get_next_batch(task_id, mode):
    assert LOCAL_DATA_LOADER is not None

    if task_id not in TASK_DATA_LOADERS:
        with LOCK:
            TASK_DATA_LOADERS[task_id] = copy.deepcopy(LOCAL_DATA_LOADER)

    db = TASK_DATA_LOADERS[task_id]
    x_spt, y_spt, x_qry, y_qry = db.next(mode)
    x_spt, y_spt, x_qry, y_qry = x_spt[task_id], y_spt[task_id], x_qry[task_id], y_qry[task_id]
    return x_qry, y_qry, x_spt, y_spt


@todist.auto_init_rpc(worker_init)
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32,
    )
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    set_local_data_loader(args, device=LOCAL_DEVICE)
    todist.barrier()  # ensure that all workers have loaded the data

    # Create a vanilla PyTorch neural network.
    net = build_model(args, device=torch.device('cpu'))

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    log = []
    test(net, epoch=-1, log=log)
    for epoch in range(10):
        train(net, meta_opt, epoch=epoch, log=log)
        test(net, epoch=epoch, log=log)
        plot(log)


def args_replicator(net_rref, n_inner_iter, task_id, task_num, mode):
    del task_id
    num_workers = todist.get_world_size()
    return [
        (task_id % num_workers, (net_rref, n_inner_iter, task_id, task_num, mode), None)
        for task_id in range(task_num)
    ]


def transpose_mean_reducer(results):
    qry_losses, qry_accs = tuple(zip(*results))
    qry_loss = torch.mean(torch.stack(qry_losses))
    qry_acc = np.mean(qry_accs)
    return qry_loss, qry_acc


@todist.parallelize(partitioner=args_replicator, reducer=transpose_mean_reducer)
def inner_loop(net_rref, n_inner_iter, task_id, task_num, mode):
    device = LOCAL_DEVICE

    original_net = net_rref.to_here()
    # The local net can be shared across multiple RPC calls on the current worker
    # We need to detach the buffers to avoid sharing the same buffers across
    net = torchopt.module_clone(original_net, by='reference', detach_buffers=True, device=device)

    x_spt, y_spt, x_qry, y_qry = get_next_batch(task_id, mode)
    if device is not None:
        x_spt = x_spt.to(device)
        y_spt = y_spt.to(device)
        x_qry = x_qry.to(device)
        y_qry = y_qry.to(device)

    inner_opt = torchopt.MetaSGD(net, lr=1e-1)

    for _ in range(n_inner_iter):
        spt_logits = net(x_spt)
        spt_loss = F.cross_entropy(spt_logits, y_spt)
        inner_opt.step(spt_loss)

    qry_logits = net(x_qry)
    qry_loss = F.cross_entropy(qry_logits, y_qry).cpu()
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).float().mean().item()

    return qry_loss, qry_acc


@todist.rank_zero_only
def train(net: nn.Module, meta_opt: optim.Adam, epoch: int, log: list):
    net.train()

    db = LOCAL_DATA_LOADER
    n_train_iter = db.x_train.shape[0] // db.batchsz
    task_num = db.x_train.shape[1]

    net_rref = todist.rpc.RRef(net)
    for batch_idx in range(n_train_iter):
        start_time = time.time()

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5

        meta_opt.zero_grad()
        # Sending modules contains nn.Parameter will detach from the current computation graph
        # Here we explicitly convert the parameters to tensors with `CloneBackward`
        net_rref = todist.rpc.RRef(torchopt.module_clone(net, by='copy'))
        with todist.autograd.context() as context_id:
            qry_loss, qry_acc = inner_loop(net_rref, n_inner_iter, None, task_num, 'train')
            todist.autograd.backward(context_id, qry_loss)
            meta_opt.step()

        qry_loss = qry_loss.item()
        qry_acc = 100.0 * qry_acc
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time

        print(
            f'[Epoch {i:.2f}] Train Loss: {qry_loss:.2f} | Acc: {qry_acc:.2f} | Time: {iter_time:.2f}',
        )
        log.append(
            {
                'epoch': i,
                'loss': qry_loss,
                'acc': qry_acc,
                'mode': 'train',
                'time': time.time(),
            },
        )


@todist.rank_zero_only
def test(net, epoch, log):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()

    db = LOCAL_DATA_LOADER
    n_test_iter = db.x_test.shape[0] // db.batchsz
    task_num = db.x_train.shape[1]

    qry_losses = []
    qry_accs = []

    net_rref = todist.rpc.RRef(net)
    for _ in range(n_test_iter):
        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5

        qry_loss, qry_acc = inner_loop(net_rref, n_inner_iter, None, task_num, 'test')
        qry_losses.append(qry_loss.item())
        qry_accs.append(qry_acc)

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


@todist.rank_zero_only
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
    ax.set_ylim(85, 100)
    ax.set_title('Distributed MAML Omniglot')
    ax.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()
