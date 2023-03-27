# Copyright 2022 MetaOPT Team. All Rights Reserved.
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
# https://github.com/uber-research/learning-to-reweight-examples
# ==============================================================================
# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

import torchopt
from helpers.argument import parse_args
from helpers.model import LeNet5
from helpers.utils import get_imbalance_dataset, plot, set_seed


def run_baseline(args, mnist_train, mnist_test):
    print('Run Baseline')
    set_seed(args.seed)

    pos_ratio = args.pos_ratio
    ntrain = args.ntrain
    nval = args.nval
    ntest = args.ntest
    epoch = args.epoch

    writer = SummaryWriter('./result/baseline')
    with open('./result/baseline/config.json', 'w') as f:
        json.dump(args.__dict__, f)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_set, _, test_set = get_imbalance_dataset(
        mnist_train,
        mnist_test,
        pos_ratio=pos_ratio,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    model = LeNet5(args).to(args.device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    running_train_loss = []
    test_acc_result = []
    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(args.device), train_label.to(args.device)
            outer_loss = model.outer_loss(train_x, train_label)

            model_optimizer.zero_grad()
            outer_loss.backward()
            model_optimizer.step()

            running_train_loss.append(outer_loss.item())
            writer.add_scalar('train_loss', outer_loss.item(), step)

            if step % 10 == 0 and step > 0:
                running_train_mean = np.mean(np.array(running_train_loss))
                print(f'EPOCH: {_epoch}, BATCH: {idx}, LOSS: {running_train_mean}')
                writer.add_scalar('running_train_loss', running_train_mean, step)
                running_train_loss = []

            step += 1

        print('Beginning to Test')
        model.eval()
        train_acc = evaluate(train_loader, model, args)
        test_acc = evaluate(test_loader, model, args)
        model.train()

        writer.add_scalar('train_acc', train_acc, _epoch)
        writer.add_scalar('test_acc', test_acc, _epoch)
        test_acc_result.append(test_acc)
        print(f'EPOCH: {_epoch}, TRAIN_ACC: {train_acc}, TEST_ACC: {test_acc}')
    return test_acc_result


def run_L2R(args, mnist_train, mnist_test):
    print('Run L2R')
    set_seed(args.seed)

    pos_ratio = args.pos_ratio
    ntrain = args.ntrain
    nval = args.nval
    ntest = args.ntest
    epoch = args.epoch

    writer = SummaryWriter('./result/l2r/log')
    with open('./result/l2r/config.json', 'w') as f:
        json.dump(args.__dict__, f)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_set, val_set, test_set = get_imbalance_dataset(
        mnist_train,
        mnist_test,
        pos_ratio=pos_ratio,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    model = LeNet5(args).to(args.device)
    model_optimizer = torchopt.MetaSGD(model, lr=args.lr)
    real_model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    running_valid_loss = []
    valid = iter(valid_loader)
    running_train_loss = []
    test_acc_result = []
    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            try:
                valid_x, valid_label = valid.next()
            except BaseException:
                valid = iter(valid_loader)
                valid_x, valid_label = valid.next()
            train_x, train_label, valid_x, valid_label = (
                train_x.to(args.device),
                train_label.to(args.device),
                valid_x.to(args.device),
                valid_label.to(args.device),
            )

            # reset meta-parameter weights
            model.reset_meta(size=train_x.size(0))

            net_state_dict = torchopt.extract_state_dict(model)
            optim_state_dict = torchopt.extract_state_dict(model_optimizer)

            for _ in range(1):
                inner_loss = model.inner_loss(train_x, train_label)
                model_optimizer.step(inner_loss)

            # calculate outer_loss, derive meta-gradient and normalize
            outer_loss = model.outer_loss(valid_x, valid_label)
            model.meta_weights = -torch.autograd.grad(outer_loss, model.meta_weights)[0]
            model.meta_weights = torch.nn.ReLU()(model.meta_weights)
            model.normalise()

            # log loss
            running_valid_loss.append(outer_loss.item())
            writer.add_scalar('validation_loss', outer_loss.item(), step)

            # reset the model and model optimizer
            torchopt.recover_state_dict(model, net_state_dict)
            torchopt.recover_state_dict(model_optimizer, optim_state_dict)

            # reuse inner_adapt to conduct real update based on learned meta weights
            inner_loss = model.inner_loss(train_x, train_label)
            for _ in range(1):
                inner_loss = model.inner_loss(train_x, train_label)
                real_model_optimizer.zero_grad()
                inner_loss.backward()
                real_model_optimizer.step()

            running_train_loss.append(inner_loss.item())
            writer.add_scalar('weighted_train_loss', inner_loss.item(), step)

            if step % 10 == 0 and step > 0:
                running_valid_mean = np.mean(np.array(running_valid_loss))
                running_train_mean = np.mean(np.array(running_train_loss))
                print(
                    'EPOCH: {}, BATCH: {}, WEIGHTED_TRAIN_LOSS: {}, VALID_LOSS: {}'.format(
                        _epoch,
                        idx,
                        running_train_mean,
                        running_valid_mean,
                    ),
                )
                running_valid_loss = []
                running_train_loss = []
                writer.add_scalar('running_valid_loss', running_valid_mean, step)
                writer.add_scalar('running_train_loss', running_train_mean, step)

            step += 1

        print('Beginning to Test')
        model.eval()
        train_acc = evaluate(train_loader, model, args)
        test_acc = evaluate(test_loader, model, args)
        model.train()

        writer.add_scalar('train_acc', train_acc, _epoch)
        writer.add_scalar('test_acc', test_acc, _epoch)
        test_acc_result.append(test_acc)
        print(f'EPOCH: {_epoch}, TRAIN_ACC: {train_acc}, TEST_ACC: {test_acc}')
    return test_acc_result


def evaluate(data_loader, model, args):
    running_accuracy = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, outputs = data
            inputs, outputs = inputs.to(args.device), outputs.to(args.device)
            predicted = model(inputs)
            predicted[predicted >= 0.5] = 1.0
            predicted[predicted < 0.5] = 0.0
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

    accuracy = running_accuracy / total
    return accuracy


def main():
    mnist_train = MNIST(root='./helper/mnist_data', download=True, train=True)
    mnist_test = MNIST(root='./helper/mnist_data', download=True, train=False)
    args = parse_args()

    assert args.algo in ['baseline', 'l2r', 'both']
    if args.algo == 'baseline':
        run_baseline(args, mnist_train, mnist_test)
    elif args.algo == 'l2r':
        run_L2R(args, mnist_train, mnist_test)
    else:
        baseline_test_acc = run_baseline(args, mnist_train, mnist_test)
        args = parse_args()
        l2r_test_acc = run_L2R(args, mnist_train, mnist_test)
        plot(baseline_test_acc, l2r_test_acc)


if __name__ == '__main__':
    main()
