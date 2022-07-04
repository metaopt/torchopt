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

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torchvision import models

import torchopt


class HighLevelInplace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = models.resnet18()
        cls.model_backup = copy.deepcopy(cls.model)
        cls.model_ref = copy.deepcopy(cls.model)

        cls.batch_size = 2
        cls.dataset = data.TensorDataset(torch.randn(2, 3, 224, 224), torch.randint(0, 1000, (2,)))
        cls.loader = data.DataLoader(cls.dataset, cls.batch_size, False)

        cls.lr = 1e0
        cls.max_norm = 10.

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = copy.deepcopy(self.model_backup)
        self.model_ref = copy.deepcopy(self.model_backup)

    def test_sgd(self) -> None:
        chain = torchopt.combine.chain(
            torchopt.clip.clip_grad_norm(max_norm=self.max_norm),
            torchopt.sgd(lr=self.lr)
        )
        optim = torchopt.Optimizer(self.model.parameters(), chain)
        optim_ref = torch.optim.SGD(self.model_ref.parameters(), self.lr)
        for xs, ys in self.loader:
            pred = self.model(xs)
            pred_ref = self.model_ref(xs)
            loss = F.cross_entropy(pred, ys)
            loss_ref = F.cross_entropy(pred_ref, ys)
            optim.zero_grad()
            loss.backward()
            optim.step()
            optim_ref.zero_grad()
            loss_ref.backward()
            clip_grad_norm_(self.model_ref.parameters(), max_norm=self.max_norm)
            optim_ref.step()

        with torch.no_grad():
            for p, p_ref in zip(self.model.parameters(), self.model_ref.parameters()):
                mse = F.mse_loss(p, p_ref)
                self.assertAlmostEqual(float(mse), 0)
            for b, b_ref in zip(self.model.buffers(), self.model_ref.buffers()):
                b = b.float() if not b.is_floating_point() else b
                b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
                mse = F.mse_loss(b, b_ref)
                self.assertAlmostEqual(float(mse), 0)


if __name__ == '__main__':
    unittest.main()
