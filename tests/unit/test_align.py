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


class HighLevelLowLevelAlignment(unittest.TestCase):
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
        optim = torchopt.MetaAdamW(self.model, self.lr)
        model_ref, params_ref, buffers_ref = functorch.make_functional_with_buffers(self.model_ref)
        optim_ref = torchopt.adamw(self.lr)
        optim_state = optim_ref.init(params_ref)
        for xs, ys in self.loader:
            pred = self.model(xs)
            pred_ref = model_ref(params_ref, buffers_ref, xs)
            loss = F.cross_entropy(pred, ys)
            loss_ref = F.cross_entropy(pred_ref, ys)

            optim.step(loss)

            grad = torch.autograd.grad(loss_ref, params_ref)
            updates, optim_state = optim_ref.update(
                grad, optim_state, inplace=False, params=params_ref
            )
            params_ref = torchopt.apply_updates(params_ref, updates)

        with torch.no_grad():
            for p, p_ref in zip(self.model.parameters(), params_ref):
                mse = F.mse_loss(p, p_ref)
                self.assertAlmostEqual(float(mse), 0)
            for b, b_ref in zip(self.model.buffers(), buffers_ref):
                b = b.float() if not b.is_floating_point() else b
                b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
                mse = F.mse_loss(b, b_ref)
                self.assertAlmostEqual(float(mse), 0)


if __name__ == '__main__':
    unittest.main()
