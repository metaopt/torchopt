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

#include "include/adam_op/adam_op_impl_cpu.h"

#include <omp.h>
#include <torch/extension.h>

#include <vector>

#include "include/utils.h"

namespace torchopt {
using std::size_t;

namespace adam_op {

constexpr size_t MIN_NUMEL_USE_OMP = 1000;

template <typename scalar_t, typename other_t>
void adamForwardInplaceCPUKernel(const other_t b1,
                                 const other_t inv_one_minus_pow_b1,
                                 const other_t b2,
                                 const other_t inv_one_minus_pow_b2,
                                 const other_t eps,
                                 const other_t eps_root,
                                 const size_t n,
                                 scalar_t *__restrict__ updates_ptr,
                                 scalar_t *__restrict__ mu_ptr,
                                 scalar_t *__restrict__ nu_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t updates = updates_ptr[tid];
    const scalar_t mu = mu_ptr[tid];
    const scalar_t nu = nu_ptr[tid];

    const scalar_t mu_out = b1 * mu + (1 - b1) * updates;
    const scalar_t nu_out = b2 * nu + (1 - b2) * updates * updates;
    const scalar_t mu_hat = mu_out * inv_one_minus_pow_b1;
    const scalar_t nu_hat = nu_out * inv_one_minus_pow_b2;

    const scalar_t updates_out = mu_hat / (sqrt(nu_hat + eps_root) + eps);

    mu_ptr[tid] = mu_out;
    nu_ptr[tid] = nu_out;
    updates_ptr[tid] = updates_out;
  }
}

TensorArray<3> adamForwardInplaceCPU(const torch::Tensor &updates,
                                     const torch::Tensor &mu,
                                     const torch::Tensor &nu,
                                     const pyfloat_t b1,
                                     const pyfloat_t b2,
                                     const pyfloat_t eps,
                                     const pyfloat_t eps_root,
                                     const pyuint_t count) {
  using other_t = pyfloat_t;
  const other_t inv_one_minus_pow_b1 = 1 / (1 - std::pow(b1, count));
  const other_t inv_one_minus_pow_b2 = 1 / (1 - std::pow(b2, count));

  const size_t n = getTensorPlainSize(updates);
  AT_DISPATCH_SCALAR_TYPES(updates.scalar_type(), "adamForwardInplaceCPU", ([&] {
                             adamForwardInplaceCPUKernel<scalar_t, scalar_t>(
                                 scalar_t(b1),
                                 scalar_t(inv_one_minus_pow_b1),
                                 scalar_t(b2),
                                 scalar_t(inv_one_minus_pow_b2),
                                 scalar_t(eps),
                                 scalar_t(eps_root),
                                 n,
                                 updates.data_ptr<scalar_t>(),
                                 mu.data_ptr<scalar_t>(),
                                 nu.data_ptr<scalar_t>());
                           }));
  return TensorArray<3>{updates, mu, nu};
}

template <typename scalar_t, typename other_t>
void adamForwardMuCPUKernel(const scalar_t *__restrict__ updates_ptr,
                            const scalar_t *__restrict__ mu_ptr,
                            const other_t b1,
                            const size_t n,
                            scalar_t *__restrict__ mu_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t updates = updates_ptr[tid];
    const scalar_t mu = mu_ptr[tid];
    const scalar_t mu_out = b1 * mu + (1 - b1) * updates;
    mu_out_ptr[tid] = mu_out;
  }
}

torch::Tensor adamForwardMuCPU(const torch::Tensor &updates,
                               const torch::Tensor &mu,
                               const pyfloat_t b1) {
  auto mu_out = torch::empty_like(mu);

  const size_t n = getTensorPlainSize(updates);
  AT_DISPATCH_SCALAR_TYPES(updates.scalar_type(), "adamForwardMuCPU", ([&] {
                             adamForwardMuCPUKernel<scalar_t, scalar_t>(
                                 updates.data_ptr<scalar_t>(),
                                 mu.data_ptr<scalar_t>(),
                                 scalar_t(b1),
                                 n,
                                 mu_out.data_ptr<scalar_t>());
                           }));
  return mu_out;
}

template <typename scalar_t, typename other_t>
void adamForwardNuCPUKernel(const scalar_t *__restrict__ updates_ptr,
                            const scalar_t *__restrict__ nu_ptr,
                            const other_t b2,
                            const size_t n,
                            scalar_t *__restrict__ nu_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t updates = updates_ptr[tid];
    const scalar_t nu = nu_ptr[tid];

    const scalar_t nu_out = b2 * nu + (1 - b2) * updates * updates;
    nu_out_ptr[tid] = nu_out;
  }
}

torch::Tensor adamForwardNuCPU(const torch::Tensor &updates,
                               const torch::Tensor &nu,
                               const pyfloat_t b2) {
  auto nu_out = torch::empty_like(nu);

  const size_t n = getTensorPlainSize(updates);
  AT_DISPATCH_SCALAR_TYPES(updates.scalar_type(), "adamForwardNuCPU", ([&] {
                             adamForwardNuCPUKernel<scalar_t, scalar_t>(
                                 updates.data_ptr<scalar_t>(),
                                 nu.data_ptr<scalar_t>(),
                                 scalar_t(b2),
                                 n,
                                 nu_out.data_ptr<scalar_t>());
                           }));
  return nu_out;
}

template <typename scalar_t, typename other_t>
void adamForwardUpdatesCPUKernel(const scalar_t *__restrict__ new_mu_ptr,
                                 const scalar_t *__restrict__ new_nu_ptr,
                                 const other_t inv_one_minus_pow_b1,
                                 const other_t inv_one_minus_pow_b2,
                                 const other_t eps,
                                 const other_t eps_root,
                                 const size_t n,
                                 scalar_t *__restrict__ updates_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t new_mu = new_mu_ptr[tid];
    const scalar_t new_nu = new_nu_ptr[tid];
    const scalar_t mu_hat = new_mu * inv_one_minus_pow_b1;
    const scalar_t nu_hat = new_nu * inv_one_minus_pow_b2;
    updates_out_ptr[tid] = mu_hat / (sqrt(nu_hat + eps_root) + eps);
  }
}

torch::Tensor adamForwardUpdatesCPU(const torch::Tensor &new_mu,
                                    const torch::Tensor &new_nu,
                                    const pyfloat_t b1,
                                    const pyfloat_t b2,
                                    const pyfloat_t eps,
                                    const pyfloat_t eps_root,
                                    const pyuint_t count) {
  using other_t = pyfloat_t;
  const other_t inv_one_minus_pow_b1 = 1 / (1 - std::pow(b1, count));
  const other_t inv_one_minus_pow_b2 = 1 / (1 - std::pow(b2, count));

  auto updates_out = torch::empty_like(new_mu);

  const size_t n = getTensorPlainSize(new_mu);
  AT_DISPATCH_SCALAR_TYPES(new_mu.scalar_type(), "adamForwardUpdatesCPU", ([&] {
                             adamForwardUpdatesCPUKernel<scalar_t, scalar_t>(
                                 new_mu.data_ptr<scalar_t>(),
                                 new_nu.data_ptr<scalar_t>(),
                                 scalar_t(inv_one_minus_pow_b1),
                                 scalar_t(inv_one_minus_pow_b2),
                                 scalar_t(eps),
                                 scalar_t(eps_root),
                                 n,
                                 updates_out.data_ptr<scalar_t>());
                           }));
  return updates_out;
}

template <typename scalar_t, typename other_t>
void adamBackwardMuCPUKernel(const scalar_t *__restrict__ dmu_ptr,
                             const other_t b1,
                             const size_t n,
                             scalar_t *__restrict__ dupdates_out_ptr,
                             scalar_t *__restrict__ dmu_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t dmu = dmu_ptr[tid];

    dupdates_out_ptr[tid] = (1 - b1) * dmu;
    dmu_out_ptr[tid] = b1 * dmu;
  }
}

TensorArray<2> adamBackwardMuCPU(const torch::Tensor &dmu,
                                 const torch::Tensor &updates,
                                 const torch::Tensor &mu,
                                 const pyfloat_t b1) {
  auto dupdates_out = torch::empty_like(updates);
  auto dmu_out = torch::empty_like(mu);

  const size_t n = getTensorPlainSize(dmu);
  AT_DISPATCH_SCALAR_TYPES(dmu.scalar_type(), "adamBackwardMuCPU", ([&] {
                             adamBackwardMuCPUKernel<scalar_t, scalar_t>(
                                 dmu.data_ptr<scalar_t>(),
                                 scalar_t(b1),
                                 n,
                                 dupdates_out.data_ptr<scalar_t>(),
                                 dmu_out.data_ptr<scalar_t>());
                           }));
  return TensorArray<2>{std::move(dupdates_out), std::move(dmu_out)};
}

template <typename scalar_t, typename other_t>
void adamBackwardNuCPUKernel(const scalar_t *__restrict__ dnu_ptr,
                             const scalar_t *__restrict__ updates_ptr,
                             const other_t b2,
                             const size_t n,
                             scalar_t *__restrict__ dupdates_out_ptr,
                             scalar_t *__restrict__ dnu_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t dnu = dnu_ptr[tid];
    const scalar_t updates = updates_ptr[tid];

    dupdates_out_ptr[tid] = 2 * (1 - b2) * updates * dnu;
    dnu_out_ptr[tid] = b2 * dnu;
  }
}

TensorArray<2> adamBackwardNuCPU(const torch::Tensor &dnu,
                                 const torch::Tensor &updates,
                                 const torch::Tensor &nu,
                                 const pyfloat_t b2) {
  auto dupdates_out = torch::empty_like(updates);
  auto dnu_out = torch::empty_like(nu);

  const size_t n = getTensorPlainSize(dnu);
  AT_DISPATCH_SCALAR_TYPES(dnu.scalar_type(), "adamForwardNuCPU", ([&] {
                             adamBackwardNuCPUKernel<scalar_t, scalar_t>(
                                 dnu.data_ptr<scalar_t>(),
                                 updates.data_ptr<scalar_t>(),
                                 scalar_t(b2),
                                 n,
                                 dupdates_out.data_ptr<scalar_t>(),
                                 dnu_out.data_ptr<scalar_t>());
                           }));
  return TensorArray<2>{std::move(dupdates_out), std::move(dnu_out)};
}

template <typename scalar_t, typename other_t>
void adamBackwardUpdatesCPUKernel(const scalar_t *__restrict__ dupdates_ptr,
                                  const scalar_t *__restrict__ updates_ptr,
                                  const scalar_t *__restrict__ new_mu_ptr,
                                  const other_t one_minus_pow_b1,
                                  const other_t inv_one_minus_pow_b2,
                                  const size_t n,
                                  scalar_t *__restrict__ dnew_mu_out_ptr,
                                  scalar_t *__restrict__ dnew_nu_out_ptr) {
#pragma omp parallel for num_threads(std::min( \
    n / MIN_NUMEL_USE_OMP, static_cast <size_t>(omp_get_num_procs()))) if (n > MIN_NUMEL_USE_OMP)
  for (size_t tid = 0; tid < n; ++tid) {
    const scalar_t dupdates = dupdates_ptr[tid];
    const scalar_t updates = updates_ptr[tid];
    const scalar_t new_mu = new_mu_ptr[tid];

    if (new_mu == 0) {
      dnew_mu_out_ptr[tid] = 0;
      dnew_nu_out_ptr[tid] = 0;
      continue;
    }

    const scalar_t updates_div_new_mu = updates / new_mu;

    const scalar_t denominator = updates_div_new_mu * one_minus_pow_b1;

    dnew_mu_out_ptr[tid] = dupdates * updates_div_new_mu;
    dnew_nu_out_ptr[tid] =
        -dupdates * updates * denominator * 0.5 * inv_one_minus_pow_b2 * denominator;
  }
}

TensorArray<2> adamBackwardUpdatesCPU(const torch::Tensor &dupdates,
                                      const torch::Tensor &updates,
                                      const torch::Tensor &new_mu,
                                      const torch::Tensor &new_nu,
                                      const pyfloat_t b1,
                                      const pyfloat_t b2,
                                      const pyfloat_t eps_root,
                                      const pyuint_t count) {
  using other_t = pyfloat_t;
  const other_t one_minus_pow_b1 = 1 - std::pow(b1, count);
  const other_t inv_one_minus_pow_b2 = 1 / (1 - std::pow(b2, count) + eps_root);

  auto dmu_out = torch::empty_like(new_mu);
  auto dnu_out = torch::empty_like(new_nu);

  const size_t n = getTensorPlainSize(dupdates);
  AT_DISPATCH_SCALAR_TYPES(dupdates.scalar_type(), "adamBackwardUpdatesCPU", ([&] {
                             adamBackwardUpdatesCPUKernel<scalar_t, scalar_t>(
                                 dupdates.data_ptr<scalar_t>(),
                                 updates.data_ptr<scalar_t>(),
                                 new_mu.data_ptr<scalar_t>(),
                                 scalar_t(one_minus_pow_b1),
                                 scalar_t(inv_one_minus_pow_b2),
                                 n,
                                 dmu_out.data_ptr<scalar_t>(),
                                 dnu_out.data_ptr<scalar_t>());
                           }));
  return TensorArray<2>{std::move(dmu_out), std::move(dnu_out)};
}

}  // namespace adam_op
}  // namespace torchopt
