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

def forward_(updates, mu, nu, lr, b1, b2, eps, eps_root, count): ...


def forwardMu(updates, mu, b1): ...


def forwardNu(updates, nu, b2): ...


def forwardUpdates(new_mu, new_nu, lr, b1, b2, eps, eps_root, count): ...


def backwardMu(dmu, updates, mu, b1): ...


def backwardNu(dnu, updates, nu, b2): ...


def backwardUpdates(dupdates, updates, new_mu, new_nu, lr, b1, b2, count): ...
