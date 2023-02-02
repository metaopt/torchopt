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

#include <array>

#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif

namespace torchopt {
__forceinline__ size_t getTensorPlainSize(const torch::Tensor &tensor) {
  const auto dim = tensor.dim();
  size_t n = 1;
  for (std::decay_t<decltype(dim)> i = 0; i < dim; ++i) {
    n *= tensor.size(i);
  }
  return n;
}
}  // namespace torchopt
