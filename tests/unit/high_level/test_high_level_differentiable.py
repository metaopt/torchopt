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

import copy
import unittest

import functorch
import pytest
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import models

import torchopt


class HighLevelDifferentiable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = models.resnet18()
        cls.model_ref = copy.deepcopy(cls.model)
        cls.model_backup = copy.deepcopy(cls.model)

        cls.batch_size = 2
        cls.dataset = data.TensorDataset(torch.randn(2, 3, 224, 224), torch.randint(0, 1000, (2,)))
        cls.loader = data.DataLoader(cls.dataset, cls.batch_size, False)

        cls.lr = 1e-3

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = copy.deepcopy(self.model_backup)
        self.model_ref = copy.deepcopy(self.model_backup)

    def test_adamw(self) -> None:
        outer_optim = torch.optim.AdamW(self.model.parameters(), self.lr)
        inner_optim = torchopt.MetaAdamW(self.model, self.lr)
        model_ref, params_ref, buffers_ref = functorch.make_functional_with_buffers(self.model_ref)
        outer_optim_ref = torchopt.adamw(self.lr)
        outer_optim_state = outer_optim_ref.init(params_ref)
        inner_optim_ref = torchopt.adamw(self.lr)
        inner_optim_state = inner_optim_ref.init(params_ref)
        for xs, ys in self.loader:

            # inner step
            inner_pred = self.model(xs)
            inner_loss = F.cross_entropy(inner_pred, ys)
            inner_optim.step(inner_loss)
            # outer step
            outer_pred = self.model(xs)
            outer_loss = F.cross_entropy(outer_pred, ys)
            outer_loss.backward()
            outer_optim.step()

            inner_pred_ref = model_ref(params_ref, buffers_ref, xs)
            inner_loss_ref = F.cross_entropy(inner_pred_ref, ys)
            grad = torch.autograd.grad(inner_loss_ref, params_ref)
            inner_updates, inner_optim_state = inner_optim_ref.update(
                grad, inner_optim_state, inplace=False, params=params_ref
            )
            new_params = torchopt.apply_updates(params_ref, inner_updates)
            outer_pred_ref = model_ref(new_params, buffers_ref, xs)
            outer_loss_ref = F.cross_entropy(outer_pred_ref, ys)

            meta_grad = torch.autograd.grad(outer_loss_ref, params_ref)
            meta_updates, outer_optim_state = outer_optim_ref.update(
                meta_grad, outer_optim_state, inplace=True, params=params_ref
            )
            params_ref = torchopt.apply_updates(params_ref, meta_updates)

        with torch.no_grad():
            for p, p_ref in zip(self.model.parameters(), params_ref):
                mse = F.mse_loss(p, p_ref)
                self.assertAlmostEqual(float(mse), 0, delta=1e-5)
            for b, b_ref in zip(self.model.buffers(), buffers_ref):
                b = b.float() if not b.is_floating_point() else b
                b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
                mse = F.mse_loss(b, b_ref)
                self.assertAlmostEqual(float(mse), 0)


if __name__ == '__main__':
    unittest.main()

    # def meta_loss_fn(params_with_buffers, data):
    #     params, buffers = params_with_buffers
    #     x_s, y_s = data
    #     """Computes the loss after one step of AdamW."""
    #     inner_pred_ref = model_ref(params, buffers, x_s)
    #     inner_loss_ref = F.cross_entropy(inner_pred_ref, y_s)
    #     grad = torch.autograd.grad(inner_loss_ref, params)
    #     inner_updates, inner_optim_state = inner_optim_ref.update(
    #         grad, inner_optim_state, inplace=False, params=params
    #     )
    #     new_params = torchopt.apply_updates(params, inner_updates)
    #     outer_pred_ref = model_ref(new_params, buffers, x_s)
    #     outer_loss_ref = F.cross_entropy(outer_pred_ref, y_s)
    #     return outer_loss_ref
    # meta_grad = functorch.grad(meta_loss_fn)((params_ref, buffers_ref), (xs, ys))
