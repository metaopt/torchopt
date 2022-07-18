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

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble, grad_and_value, make_functional, vmap

import torchopt


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
# to parallelize training over models.


def make_spirals(n_samples, noise_std=0., rotations=1.):
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


def train_step_fn(weights, batch, targets, lr=0.2):

    def compute_loss(weights, batch, targets):
        output = func_model(weights, batch)
        loss = loss_fn(output, targets)
        return loss

    grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)

    # functional optimizer API is here now
    new_weights = []
    with torch.no_grad():
        for grad_weight, weight in zip(grad_weights, weights):
            new_weights.append(weight - grad_weight * lr)

    return loss, new_weights


def step4():
    global weights
    for i in range(2000):
        loss, weights = train_step_fn(weights, points, labels)
        if i % 100 == 0:
            print(loss)


def init_fn(num_models):
    models = [MLPClassifier().to(DEVICE) for _ in range(num_models)]
    _, params, _ = combine_state_for_ensemble(models)
    return params


def step6():
    parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None))
    batched_weights = init_fn(num_models=2)
    for i in range(2000):
        loss, batched_weights = parallel_train_step_fn(batched_weights, points, labels)
        if i % 200 == 0:
            print(loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Functorch Ensembled Models with TorchOpt")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="CPU or GPU ID for this process (default: 'cpu')",
    )
    args = parser.parse_args()

    DEVICE = args.device
    # Step 1: Make some spirals
    points, labels = make_spirals(100, noise_std=0.05)
    # Step 2: Define two-layer MLP and loss function
    loss_fn = nn.NLLLoss()
    # Step 3: Make the model functional(!!) and define a training function.
    # NB: this mechanism doesn't exist in PyTorch today, but we want it to:
    # https://github.com/pytorch/pytorch/issues/49171
    func_model, weights = make_functional(MLPClassifier().to(DEVICE))
    # Step 4: Let's verify this actually trains.
    # We should see the loss decrease.
    step4()
    # Step 5: We're ready for multiple models. Let's define an init_fn
    # that, given a number of models, returns to us all of the weights.

    # Step 6: Now, can we try multiple models at the same time?
    # The answer is: yes! `loss` is a 2-tuple, and we can see that the value keeps
    # on decreasing
    step6()
    # Step 7: Now, the flaw with step 6 is that we were training on the same exact
    # data. This can lead to all of the models in the ensemble overfitting in the
    # same way. The solution that http://willwhitney.com/parallel-training-jax.html
    # applies is to randomly subset the data in a way that the models do not recieve
    # exactly the same data in each training step!
    # Because the goal of this doc is to show that we can use eager-mode vmap to
    # achieve similar things as JAX, the rest of this is left as an exercise to the reader.



    # cuda_is_avail = torch.cuda.is_available()
    # print(f"cuda_is_avail: {cuda_is_avail}")
    # DEVICE = "cuda" if cuda_is_avail else "cpu"

    # learning_rate = 1.
    # batch_size = 1
    # dim = 3
    # # Ignore `params` since we'll make them in init_fn
    # func, _ = make_functional(Net(dim).to(DEVICE))
    # # Fairly certain use_accelerated_op works in my real code, but not this toy example for some reason
    # # optimizer = TorchOpt.adam(learning_rate, use_accelerated_op=cuda_is_avail)
    # optimizer = TorchOpt.adam(learning_rate)

    # class Net(nn.Module):
    #     def __init__(self, dim):
    #         super().__init__()
    #         self.fc = nn.Linear(dim, 1, bias=True)
    #         nn.init.ones_(self.fc.weight)
    #         nn.init.zeros_(self.fc.bias)

    #     def forward(self, x):
    #         return self.fc(x)

    # # We don't really need parallelized init like this, but it might be the easiest way just for
    # # matching the data format required for the par train step
    # def init_fn(dummy_parallel):
    #     # Ignore `func` since all of them are identical. Just use `func` from earlier instead
    #     _, params = make_functional(Net(dim).to(DEVICE))

    #     # print(params)
    #     opt_state = optimizer.init(params)
    #     print("opt_state before add tensor to count")
    #     print(opt_state)
    #     print()

    #     # TorchOpt doesn't give tensor types for `count` -- add that here:
    #     opt_state = (
    #         opt_state[0]._replace(
    #                 # Doesn't work -- possible this with some modification could be best
    #                 # count=transpose_stack(opt_state[0].count)

    #                 # Not clear if we want [el] or even [[el]] here (even if not ultimately written
    #                 # quite like that), which IIRC worked fine too so far
    #                 count=tuple(torch.tensor(el).to(DEVICE) for el in opt_state[0].count)
    #             ),
    #         opt_state[1]
    #     )

    #     print("opt_state after add tensor to count")
    #     print(opt_state)
    #     print()

    #     return params, opt_state

    # xs = 2 * torch.ones(batch_size, dim).to(DEVICE)
    # ys = torch.ones(batch_size).to(DEVICE)

    # def train_step_fn(params, opt_state):
    #     def compute_loss(params):
    #         pred = func(params, xs)
    #         loss = ((pred - ys) ** 2).sum()
    #         print("loss", loss)
    #         return loss

    #     # grads = torch.autograd.grad(loss, params)
    #     grads, loss = functorch.grad_and_value(compute_loss)(params)
    #     # print(grads)
    #     # print(opt_state)

    #     # print()
    #     # print("opt_state before clone")
    #     # print(opt_state)

    #     # Clone tensors to avoid error (TODO can surely avoid this better)
    #     new_opt_state0 = opt_state[0]._asdict()
    #     for k, v in new_opt_state0.items():
    #         if type(v) is tuple:
    #             new_opt_state0[k] = tuple(v_el.clone() for v_el in v)
    #     new_opt_state = (opt_state[0]._make(new_opt_state0.values()), opt_state[1])

    #     # # If we can't get rid of clone above, then we should maybe add an assert of equality
    #     # # or torch.allclose() or sth here. IIRC plain assert == worked outside of vmap, but not in vmap.
    #     # #
    #     # print()
    #     # print("opt_state after clone")
    #     # print(new_opt_state)
    #     # print()

    #     opt_state = new_opt_state

    #     print()
    #     print("opt_state:")
    #     print(opt_state)
    #     print()
    #     print()
    #     print("grads:")
    #     print(grads)
    #     print()

    #     with torch.no_grad():
    #         updates, opt_state = optimizer.update(grads, opt_state)

    #         print()
    #         print("params")
    #         print(params)
    #         print()

    #         print()
    #         print("updates")
    #         print(updates)
    #         print()

    #         # Default `inplace=True` gave me an error
    #         params = TorchOpt.apply_updates(params, updates, inplace=False)
    #         # print(params)

    #     return params, opt_state

    # # NOTE Any other `randomness` setting threw a bug. Real code can't use this as written, bc gives
    # # identical inits for each net in the ensemble
    # parallel_init_fn = functorch.vmap(init_fn, randomness='same')
    # parallel_train_step_fn = functorch.vmap(train_step_fn)

    # params, opt_states = parallel_init_fn(torch.ones(2,))

    # print("opt_states[0] before par train step fn")
    # print(opt_states[0])
    # print()
    # print([len(t) for t in opt_states[0]])
    # print()
    # for tup in opt_states[0]:
    #     print(len(tup))
    #     print([tns.shape for tns in tup])
    # params, opt_states = parallel_train_step_fn(params, opt_states)

    # print(params)