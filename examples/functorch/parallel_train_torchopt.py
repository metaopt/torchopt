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

import argparse
import math
from collections import namedtuple
from typing import Any, NamedTuple

import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble, grad_and_value, make_functional, vmap

import torchopt


def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
    ts = torch.linspace(0, 1, n_samples, device=DEVICE)
    rs = ts**0.5
    thetas = rs * rotations * 2 * math.pi
    signs = torch.randint(0, 2, (n_samples,), device=DEVICE) * 2 - 1
    labels = (signs > 0).to(torch.long).to(DEVICE)

    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return points, labels


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim=32, n_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.fc1 = nn.Linear(2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, -1)
        return x


class ParallelTrainFunctorchOriginal:
    def __init__(self, loss_fn, lr):
        self.loss_fn = loss_fn
        self.lr = lr
        self.func_model, _ = make_functional(MLPClassifier().to(DEVICE))

    def init_fn(self, num_models):
        models = [MLPClassifier().to(DEVICE) for _ in range(num_models)]
        _, batched_weights, _ = combine_state_for_ensemble(models)
        return batched_weights

    def train_step_fn(self, weights, batch, targets):
        def compute_loss(weights, batch, targets):
            output = self.func_model(weights, batch)
            loss = self.loss_fn(output, targets)
            return loss

        grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)
        # NB: PyTorch is missing a "functional optimizer API" (possibly coming soon)
        # so we are going to re-implement SGD here.
        new_weights = []
        with torch.no_grad():
            for grad_weight, weight in zip(grad_weights, weights):
                new_weights.append(weight - grad_weight * self.lr)

        return loss, new_weights

    def test_train_step_fn(self, weights, points, labels):
        for i in range(2000):
            loss, weights = self.train_step_fn(weights, points, labels)
            if i % 100 == 0:
                print(loss)

    def test_parallel_train_step_fn(self, num_models):
        parallel_train_step_fn = vmap(self.train_step_fn, in_dims=(0, None, None))
        batched_weights = self.init_fn(num_models=num_models)
        for i in range(2000):
            loss, batched_weights = parallel_train_step_fn(batched_weights, points, labels)
            if i % 200 == 0:
                print(loss)


class ParallelTrainFunctorchTorchOpt:
    def __init__(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.func_model, _ = make_functional(MLPClassifier().to(DEVICE))

    def init_fn(self, model_idx):
        _, weights = make_functional(MLPClassifier().to(DEVICE))
        opt_state = self.optimizer.init(weights)
        return weights, opt_state

    def train_step_fn(self, training_state, batch, targets):
        weights, opt_state = training_state

        def compute_loss(weights, batch, targets):
            output = self.func_model(weights, batch)
            loss = self.loss_fn(output, targets)
            return loss

        grads, loss = grad_and_value(compute_loss)(weights, batch, targets)
        # functional optimizer API is here now
        updates, new_opt_state = optimizer.update(grads, opt_state, inplace=False)
        new_weights = torchopt.apply_updates(weights, updates, inplace=False)
        return loss, (new_weights, new_opt_state)

    def test_train_step_fn(self, weights, opt_state, points, labels):
        for i in range(2000):
            loss, (weights, opt_state) = self.train_step_fn((weights, opt_state), points, labels)
            if i % 100 == 0:
                print(loss)

    def test_parallel_train_step_fn(self, num_models):
        parallel_init_fn = vmap(self.init_fn, randomness='same')
        parallel_train_step_fn = vmap(self.train_step_fn, in_dims=(0, None, None))
        weights, opt_state = parallel_init_fn(torch.ones(num_models, 1))
        for i in range(2000):
            loss, (weights, opt_states) = parallel_train_step_fn(
                (weights, opt_state), points, labels
            )
            if i % 200 == 0:
                print(loss)


if __name__ == '__main__':
    # Adapted from http://willwhitney.com/parallel-training-jax.html , which is a
    # tutorial on Model Ensembling with JAX by Will Whitney.
    #
    # The original code comes with the following citation:
    # @misc{Whitney2021Parallelizing,
    #     author = {William F. Whitney},
    #     title = { {Parallelizing neural networks on one GPU with JAX} },
    #     year = {2021},
    #     url = {http://willwhitney.com/parallel-training-jax.html},
    # }

    # GOAL: Demonstrate that it is possible to use eager-mode vmap

    parser = argparse.ArgumentParser(description='Functorch Ensembled Models with TorchOpt')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="CPU or GPU ID for this process (default: 'cpu')",
    )
    args = parser.parse_args()

    DEVICE = args.device
    # Step 1: Make some spirals
    points, labels = make_spirals(100, noise_std=0.05)
    # Step 2: Define two-layer MLP and loss function
    loss_fn = nn.NLLLoss()
    # Step 3: Make the model functional(!!) and define a training function.
    func_model, weights = make_functional(MLPClassifier().to(DEVICE))

    # original functorch implementation
    functorch_original = ParallelTrainFunctorchOriginal(loss_fn=loss_fn, lr=0.2)
    # Step 4: Let's verify this actually trains.
    # We should see the loss decrease.
    functorch_original.test_train_step_fn(weights, points, labels)
    # Step 6: Now, can we try multiple models at the same time?
    # The answer is: yes! `loss` is a 2-tuple, and we can see that the value keeps
    # on decreasing
    functorch_original.test_parallel_train_step_fn(num_models=2)

    # functorch + torchopt implementation
    optimizer = torchopt.adam(lr=0.2)
    opt_state = optimizer.init(weights)
    functorch_original = ParallelTrainFunctorchTorchOpt(loss_fn=loss_fn, optimizer=optimizer)
    # Step 4: Let's verify this actually trains.
    # We should see the loss decrease.
    functorch_original.test_train_step_fn(weights, opt_state, points, labels)
    # Step 6: Now, can we try multiple models at the same time?
    # The answer is: yes! `loss` is a 2-tuple, and we can see that the value keeps
    # on decreasing
    functorch_original.test_parallel_train_step_fn(num_models=2)

    # Step 7: Now, the flaw with step 6 is that we were training on the same exact
    # data. This can lead to all of the models in the ensemble overfitting in the
    # same way. The solution that http://willwhitney.com/parallel-training-jax.html
    # applies is to randomly subset the data in a way that the models do not recieve
    # exactly the same data in each training step!
    # Because the goal of this doc is to show that we can use eager-mode vmap to
    # achieve similar things as JAX, the rest of this is left as an exercise to the reader.
