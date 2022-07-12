// Copyright 2022 MetaOPT Team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "include/adam_op/adam_op.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/adam_op/adam_op_impl.cuh"
#include "include/adam_op/adam_op_impl.h"

namespace torchopt {
TensorArray<3> adamForwardInplace(const torch::Tensor& updates,
                                  const torch::Tensor& mu,
                                  const torch::Tensor& nu, const float b1,
                                  const float b2, const float eps,
                                  const float eps_root, const int count) {
  if (updates.device().is_cuda()) {
    return adamForwardInplaceCUDA(updates, mu, nu, b1, b2, eps, eps_root,
                                  count);
  } else if (updates.device().is_cpu()) {
    return adamForwardInplaceCPU(updates, mu, nu, b1, b2, eps, eps_root, count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}
torch::Tensor adamForwardMu(const torch::Tensor& updates,
                            const torch::Tensor& mu, const float b1) {
  if (updates.device().is_cuda()) {
    return adamForwardMuCUDA(updates, mu, b1);
  } else if (updates.device().is_cpu()) {
    return adamForwardMuCPU(updates, mu, b1);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

torch::Tensor adamForwardNu(const torch::Tensor& updates,
                            const torch::Tensor& nu, const float b2) {
  if (updates.device().is_cuda()) {
    return adamForwardNuCUDA(updates, nu, b2);
  } else if (updates.device().is_cpu()) {
    return adamForwardNuCPU(updates, nu, b2);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

torch::Tensor adamForwardUpdates(const torch::Tensor& new_mu,
                                 const torch::Tensor& new_nu, const float b1,
                                 const float b2, const float eps,
                                 const float eps_root, const int count) {
  if (new_mu.device().is_cuda()) {
    return adamForwardUpdatesCUDA(new_mu, new_nu, b1, b2, eps, eps_root, count);
  } else if (new_mu.device().is_cpu()) {
    return adamForwardUpdatesCPU(new_mu, new_nu, b1, b2, eps, eps_root, count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardMu(const torch::Tensor& dmu,
                              const torch::Tensor& updates,
                              const torch::Tensor& mu, const float b1) {
  if (dmu.device().is_cuda()) {
    return adamBackwardMuCUDA(dmu, updates, mu, b1);
  } else if (dmu.device().is_cpu()) {
    return adamBackwardMuCPU(dmu, updates, mu, b1);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardNu(const torch::Tensor& dnu,
                              const torch::Tensor& updates,
                              const torch::Tensor& nu, const float b2) {
  if (dnu.device().is_cuda()) {
    return adamBackwardNuCUDA(dnu, updates, nu, b2);
  } else if (dnu.device().is_cpu()) {
    return adamBackwardNuCPU(dnu, updates, nu, b2);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardUpdates(const torch::Tensor& dupdates,
                                   const torch::Tensor& updates,
                                   const torch::Tensor& new_mu,
                                   const torch::Tensor& new_nu, const float b1,
                                   const float b2, const int count) {
  if (dupdates.device().is_cuda()) {
    return adamBackwardUpdatesCUDA(dupdates, updates, new_mu, new_nu, b1, b2,
                                   count);
  } else if (dupdates.device().is_cpu()) {
    return adamBackwardUpdatesCPU(dupdates, updates, new_mu, new_nu, b1, b2,
                                  count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}
}  // namespace torchopt

PYBIND11_MODULE(adam_op, m) {
  m.def("forward_", &torchopt::adamForwardInplace);
  m.def("forwardMu", &torchopt::adamForwardMu);
  m.def("forwardNu", &torchopt::adamForwardNu);
  m.def("forwardUpdates", &torchopt::adamForwardUpdates);
  m.def("backwardMu", &torchopt::adamBackwardMu);
  m.def("backwardNu", &torchopt::adamBackwardNu);
  m.def("backwardUpdates", &torchopt::adamBackwardUpdates);
}
