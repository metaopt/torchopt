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
#include <cstddef>

using pyfloat_t = double;
using pyuint_t = std::size_t;

#if defined(USE_FP16)
#define AT_DISPATCH_SCALAR_TYPES(...) AT_DISPATCH_FLOATING_TYPES(__VA_ARGS__)
#else
#define AT_DISPATCH_SCALAR_TYPES(...) \
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, __VA_ARGS__)
#endif

#if defined(USE_FP16) && defined(CUDA_HAS_FP16)
#define AT_DISPATCH_SCALAR_TYPES_CUDA(...) AT_DISPATCH_SCALAR_TYPES(__VA_ARGS__)
#else
#define AT_DISPATCH_SCALAR_TYPES_CUDA(...) AT_DISPATCH_FLOATING_TYPES(__VA_ARGS__)
#endif

namespace torchopt {
template <size_t N>
using TensorArray = std::array<torch::Tensor, N>;
}
