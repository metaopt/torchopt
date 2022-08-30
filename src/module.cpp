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
// =============================================================================

#include <pybind11/pybind11.h>

#include "include/adam_op/adam_op.h"

PYBIND11_MODULE(_C, mod) {
  py::module m = mod.def_submodule("adam_op", "Adam Ops");
  m.def("forward_", &torchopt::adam_op::adamForwardInplace);
  m.def("forwardMu", &torchopt::adam_op::adamForwardMu);
  m.def("forwardNu", &torchopt::adam_op::adamForwardNu);
  m.def("forwardUpdates", &torchopt::adam_op::adamForwardUpdates);
  m.def("backwardMu", &torchopt::adam_op::adamBackwardMu);
  m.def("backwardNu", &torchopt::adam_op::adamBackwardNu);
  m.def("backwardUpdates", &torchopt::adam_op::adamBackwardUpdates);
}
