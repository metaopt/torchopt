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

#include "include/adam_op/adam_op.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/adam_op/adam_op_impl_cpu.h"
#if defined(__USE_CUDA__)
#include "include/adam_op/adam_op_impl_cuda.cuh"
#endif

namespace torchopt {

namespace py = pybind11;

namespace adam_op {

TensorArray<3> adamForwardInplace(const torch::Tensor &updates,
                                  const torch::Tensor &mu,
                                  const torch::Tensor &nu,
                                  const pyfloat_t b1,
                                  const pyfloat_t b2,
                                  const pyfloat_t eps,
                                  const pyfloat_t eps_root,
                                  const pyuint_t count) {
#if defined(__USE_CUDA__)
  if (updates.device().is_cuda()) {
    return adamForwardInplaceCUDA(updates, mu, nu, b1, b2, eps, eps_root, count);
  }
#endif
  if (updates.device().is_cpu()) {
    return adamForwardInplaceCPU(updates, mu, nu, b1, b2, eps, eps_root, count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

torch::Tensor adamForwardMu(const torch::Tensor &updates,
                            const torch::Tensor &mu,
                            const pyfloat_t b1) {
#if defined(__USE_CUDA__)
  if (updates.device().is_cuda()) {
    return adamForwardMuCUDA(updates, mu, b1);
  }
#endif
  if (updates.device().is_cpu()) {
    return adamForwardMuCPU(updates, mu, b1);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

torch::Tensor adamForwardNu(const torch::Tensor &updates,
                            const torch::Tensor &nu,
                            const pyfloat_t b2) {
#if defined(__USE_CUDA__)
  if (updates.device().is_cuda()) {
    return adamForwardNuCUDA(updates, nu, b2);
  }
#endif
  if (updates.device().is_cpu()) {
    return adamForwardNuCPU(updates, nu, b2);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

torch::Tensor adamForwardUpdates(const torch::Tensor &new_mu,
                                 const torch::Tensor &new_nu,
                                 const pyfloat_t b1,
                                 const pyfloat_t b2,
                                 const pyfloat_t eps,
                                 const pyfloat_t eps_root,
                                 const pyuint_t count) {
#if defined(__USE_CUDA__)
  if (new_mu.device().is_cuda()) {
    return adamForwardUpdatesCUDA(new_mu, new_nu, b1, b2, eps, eps_root, count);
  }
#endif
  if (new_mu.device().is_cpu()) {
    return adamForwardUpdatesCPU(new_mu, new_nu, b1, b2, eps, eps_root, count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardMu(const torch::Tensor &dmu,
                              const torch::Tensor &updates,
                              const torch::Tensor &mu,
                              const pyfloat_t b1) {
#if defined(__USE_CUDA__)
  if (dmu.device().is_cuda()) {
    return adamBackwardMuCUDA(dmu.contiguous(), updates, mu, b1);
  }
#endif
  if (dmu.device().is_cpu()) {
    return adamBackwardMuCPU(dmu.contiguous(), updates, mu, b1);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardNu(const torch::Tensor &dnu,
                              const torch::Tensor &updates,
                              const torch::Tensor &nu,
                              const pyfloat_t b2) {
#if defined(__USE_CUDA__)
  if (dnu.device().is_cuda()) {
    return adamBackwardNuCUDA(dnu.contiguous(), updates, nu, b2);
  }
#endif
  if (dnu.device().is_cpu()) {
    return adamBackwardNuCPU(dnu.contiguous(), updates, nu, b2);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

TensorArray<2> adamBackwardUpdates(const torch::Tensor &dupdates,
                                   const torch::Tensor &updates,
                                   const torch::Tensor &new_mu,
                                   const torch::Tensor &new_nu,
                                   const pyfloat_t b1,
                                   const pyfloat_t b2,
                                   const pyfloat_t eps_root,
                                   const pyuint_t count) {
#if defined(__USE_CUDA__)
  if (dupdates.device().is_cuda()) {
    return adamBackwardUpdatesCUDA(
        dupdates.contiguous(), updates, new_mu, new_nu, b1, b2, eps_root, count);
  }
#endif
  if (dupdates.device().is_cpu()) {
    return adamBackwardUpdatesCPU(
        dupdates.contiguous(), updates, new_mu, new_nu, b1, b2, eps_root, count);
  } else {
    throw std::runtime_error("Not implemented");
  }
}

void buildSubmodule(py::module &mod) {  // NOLINT[runtime/references]
  py::module m = mod.def_submodule("adam_op", "Adam Ops");
  m.def("forward_",
        &adamForwardInplace,
        "Adam forward inplace",
        py::arg("updates"),
        py::arg("mu"),
        py::arg("nu"),
        py::arg("b1"),
        py::arg("b2"),
        py::arg("eps"),
        py::arg("eps_root"),
        py::arg("count"));
  m.def("forward_mu",
        &adamForwardMu,
        "Adam forward mu",
        py::arg("updates"),
        py::arg("mu"),
        py::arg("b1"));
  m.def("forward_nu",
        &adamForwardNu,
        "Adam forward nu",
        py::arg("updates"),
        py::arg("nu"),
        py::arg("b2"));
  m.def("forward_updates",
        &adamForwardUpdates,
        "Adam forward updates",
        py::arg("new_mu"),
        py::arg("new_nu"),
        py::arg("b1"),
        py::arg("b2"),
        py::arg("eps"),
        py::arg("eps_root"),
        py::arg("count"));
  m.def("backward_mu",
        &adamBackwardMu,
        "Adam backward mu",
        py::arg("dmu"),
        py::arg("updates"),
        py::arg("mu"),
        py::arg("b1"));
  m.def("backward_nu",
        &adamBackwardNu,
        "Adam backward nu",
        py::arg("dnu"),
        py::arg("updates"),
        py::arg("nu"),
        py::arg("b1"));
  m.def("backward_updates",
        &adamBackwardUpdates,
        "Adam backward updates",
        py::arg("dupdates"),
        py::arg("updates"),
        py::arg("new_mu"),
        py::arg("new_nu"),
        py::arg("b1"),
        py::arg("b2"),
        py::arg("eps_root"),
        py::arg("count"));
}

}  // namespace adam_op
}  // namespace torchopt
