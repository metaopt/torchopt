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
import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import models

import torchopt
from torchopt import adam, rmsprop, sgd


class LowLevelInplace(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch.manual_seed(0)
    cls.model = models.resnet18()
    cls.model_ref = copy.deepcopy(cls.model)
    cls.model_backup = copy.deepcopy(cls.model)

    cls.batch_size = 2
    cls.dataset = data.TensorDataset(
      torch.randn(2, 3, 224, 224), torch.randint(0, 1000, (2,))
    )
    cls.loader = data.DataLoader(cls.dataset, cls.batch_size, False)

    cls.lr = 1e-3

  def setUp(self) -> None:
    torch.manual_seed(0)
    self.model = copy.deepcopy(self.model_backup)
    self.model_ref = copy.deepcopy(self.model_backup)

  def test_sgd(self) -> None:
    fun, params, buffers = functorch.make_functional_with_buffers(self.model)
    optim = sgd(self.lr)
    optim_state = optim.init(params)
    optim_ref = torch.optim.SGD(self.model_ref.parameters(), self.lr)

    for xs, ys in self.loader:
      pred = fun(params, buffers, xs)
      pred_ref = self.model_ref(xs)
      loss = F.cross_entropy(pred, ys)
      loss_ref = F.cross_entropy(pred_ref, ys)

      grad = torch.autograd.grad(loss, params)
      updates, optim_state = optim.update(grad, optim_state)
      params = torchopt.apply_updates(params, updates)

      optim_ref.zero_grad()
      loss_ref.backward()
      optim_ref.step()

    with torch.no_grad():
      for p, p_ref in zip(params, self.model_ref.parameters()):
        mse = F.mse_loss(p, p_ref)
        self.assertAlmostEqual(float(mse), 0)
      for b, b_ref in zip(buffers, self.model_ref.buffers()):
        b = b.float() if not b.is_floating_point() else b
        b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
        mse = F.mse_loss(b, b_ref)
        self.assertAlmostEqual(float(mse), 0)

  def test_adam(self) -> None:
    fun, params, buffers = functorch.make_functional_with_buffers(self.model)
    optim = adam(self.lr)
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(self.model_ref.parameters(), self.lr)
    for xs, ys in self.loader:
      pred = fun(params, buffers, xs)
      pred_ref = self.model_ref(xs)
      loss = F.cross_entropy(pred, ys)
      loss_ref = F.cross_entropy(pred_ref, ys)

      grad = torch.autograd.grad(loss, params)
      updates, optim_state = optim.update(grad, optim_state)
      params = torchopt.apply_updates(params, updates)

      optim_ref.zero_grad()
      loss_ref.backward()
      optim_ref.step()
    with torch.no_grad():
      for p, p_ref in zip(params, self.model_ref.parameters()):
        mse = F.mse_loss(p, p_ref)
        self.assertAlmostEqual(float(mse), 0)
      for b, b_ref in zip(buffers, self.model_ref.buffers()):
        b = b.float() if not b.is_floating_point() else b
        b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
        mse = F.mse_loss(b, b_ref)
        self.assertAlmostEqual(float(mse), 0)

  def test_accelerated_adam_cpu(self) -> None:
    self.model
    self.model_ref
    fun, params, buffers = functorch.make_functional_with_buffers(self.model)
    optim = adam(self.lr, use_accelerated_op=True)
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(self.model_ref.parameters(), self.lr)
    for xs, ys in self.loader:
      xs = xs
      ys = ys
      pred = fun(params, buffers, xs)
      pred_ref = self.model_ref(xs)
      loss = F.cross_entropy(pred, ys)
      loss_ref = F.cross_entropy(pred_ref, ys)

      grad = torch.autograd.grad(loss, params)
      updates, optim_state = optim.update(grad, optim_state)
      params = torchopt.apply_updates(params, updates)

      optim_ref.zero_grad()
      loss_ref.backward()
      optim_ref.step()
    with torch.no_grad():
      for p, p_ref in zip(params, self.model_ref.parameters()):
        mse = F.mse_loss(p, p_ref)
        self.assertAlmostEqual(float(mse), 0)
      for b, b_ref in zip(buffers, self.model_ref.buffers()):
        b = b.float() if not b.is_floating_point() else b
        b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
        mse = F.mse_loss(b, b_ref)
        self.assertAlmostEqual(float(mse), 0)

  def test_accelerated_adam_cuda(self) -> None:
    self.model.cuda()
    self.model_ref.cuda()
    fun, params, buffers = functorch.make_functional_with_buffers(self.model)
    optim = adam(self.lr, use_accelerated_op=True)
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(self.model_ref.parameters(), self.lr)
    for xs, ys in self.loader:
      xs = xs.cuda()
      ys = ys.cuda()
      pred = fun(params, buffers, xs)
      pred_ref = self.model_ref(xs)
      loss = F.cross_entropy(pred, ys)
      loss_ref = F.cross_entropy(pred_ref, ys)

      grad = torch.autograd.grad(loss, params)
      updates, optim_state = optim.update(grad, optim_state)
      params = torchopt.apply_updates(params, updates)

      optim_ref.zero_grad()
      loss_ref.backward()
      optim_ref.step()
    with torch.no_grad():
      for p, p_ref in zip(params, self.model_ref.parameters()):
        mse = F.mse_loss(p, p_ref)
        self.assertAlmostEqual(float(mse), 0)
      for b, b_ref in zip(buffers, self.model_ref.buffers()):
        b = b.float() if not b.is_floating_point() else b
        b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
        mse = F.mse_loss(b, b_ref)
        self.assertAlmostEqual(float(mse), 0)

  def test_rmsprop(self) -> None:
    fun, params, buffers = functorch.make_functional_with_buffers(self.model)
    optim = rmsprop(
      self.lr, decay=0.99
    )  # pytorch uses 0.99 as the default value
    optim_state = optim.init(params)
    optim_ref = torch.optim.RMSprop(self.model_ref.parameters(), self.lr)
    for xs, ys in self.loader:
      pred = fun(params, buffers, xs)
      pred_ref = self.model_ref(xs)
      loss = F.cross_entropy(pred, ys)
      loss_ref = F.cross_entropy(pred_ref, ys)

      grad = torch.autograd.grad(loss, params)
      updates, optim_state = optim.update(grad, optim_state)
      params = torchopt.apply_updates(params, updates)

      optim_ref.zero_grad()
      loss_ref.backward()
      optim_ref.step()
    with torch.no_grad():
      for p, p_ref in zip(params, self.model_ref.parameters()):
        mse = F.mse_loss(p, p_ref)
        self.assertAlmostEqual(
          float(mse), 0, delta=1e-4
        )  # Optax and pytorch have different implementation
      for b, b_ref in zip(buffers, self.model_ref.buffers()):
        b = b.float() if not b.is_floating_point() else b
        b_ref = b_ref.float() if not b_ref.is_floating_point() else b_ref
        mse = F.mse_loss(b, b_ref)
        self.assertAlmostEqual(float(mse), 0)


if __name__ == '__main__':
  unittest.main()
