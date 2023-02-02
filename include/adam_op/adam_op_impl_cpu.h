// Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
// =============================================================================

#pragma once

#include <torch/extension.h>

#include <vector>

#include "include/common.h"

namespace torchopt {
namespace adam_op {
TensorArray<3> adamForwardInplaceCPU(const torch::Tensor &updates,
                                     const torch::Tensor &mu,
                                     const torch::Tensor &nu,
                                     const pyfloat_t b1,
                                     const pyfloat_t b2,
                                     const pyfloat_t eps,
                                     const pyfloat_t eps_root,
                                     const pyuint_t count);

torch::Tensor adamForwardMuCPU(const torch::Tensor &updates,
                               const torch::Tensor &mu,
                               const pyfloat_t b1);

torch::Tensor adamForwardNuCPU(const torch::Tensor &updates,
                               const torch::Tensor &nu,
                               const pyfloat_t b2);

torch::Tensor adamForwardUpdatesCPU(const torch::Tensor &new_mu,
                                    const torch::Tensor &new_nu,
                                    const pyfloat_t b1,
                                    const pyfloat_t b2,
                                    const pyfloat_t eps,
                                    const pyfloat_t eps_root,
                                    const pyuint_t count);

TensorArray<2> adamBackwardMuCPU(const torch::Tensor &dmu,
                                 const torch::Tensor &updates,
                                 const torch::Tensor &mu,
                                 const pyfloat_t b1);

TensorArray<2> adamBackwardNuCPU(const torch::Tensor &dnu,
                                 const torch::Tensor &updates,
                                 const torch::Tensor &nu,
                                 const pyfloat_t b2);

TensorArray<2> adamBackwardUpdatesCPU(const torch::Tensor &dupdates,
                                      const torch::Tensor &updates,
                                      const torch::Tensor &new_mu,
                                      const torch::Tensor &new_nu,
                                      const pyfloat_t b1,
                                      const pyfloat_t b2,
                                      const pyfloat_t eps_root,
                                      const pyuint_t count);
}  // namespace adam_op
}  // namespace torchopt
