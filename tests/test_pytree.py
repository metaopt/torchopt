# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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

import torch

import helpers
from torchopt import pytree


tree_a = (torch.randn(20, 10), torch.randn(20))
tree_b = (torch.randn(20, 10), torch.randn(20))

tree_a_dict = (
    torch.tensor(1.0),
    {'k1': torch.tensor(1.0), 'k2': (torch.tensor(1.0), torch.tensor(1.0))},
    torch.tensor(1.0),
)
tree_b_dict = (
    torch.tensor(1.0),
    {'k1': torch.tensor(2.0), 'k2': (torch.tensor(3.0), torch.tensor(4.0))},
    torch.tensor(5.0),
)

tensor_a = torch.randn(20)
tensor_b = torch.randn(20)


def test_tree_flatten_as_tuple() -> None:
    expected_leaves, expected_treespec = (tensor_a,), pytree.tree_structure(tensor_a)
    actual_leaves, actual_treespec = pytree.tree_flatten_as_tuple(tensor_a)
    assert actual_leaves == expected_leaves
    assert actual_treespec == expected_treespec

    leaves_a, treespec_a = pytree.tree_flatten(tree_a)
    expected_leaves, expected_treespec = tuple(leaves_a), treespec_a
    actual_leaves, actual_treespec = pytree.tree_flatten_as_tuple(tree_a)
    assert actual_leaves == expected_leaves
    assert actual_treespec == expected_treespec


def test_tree_pos() -> None:
    expected = +tensor_a
    actual = pytree.tree_pos(tensor_a)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (+tree_a[0], +tree_a[1])
    actual = pytree.tree_pos(tree_a)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_neg() -> None:
    expected = -tensor_a
    actual = pytree.tree_neg(tensor_a)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (-tree_a[0], -tree_a[1])
    actual = pytree.tree_neg(tree_a)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_add() -> None:
    expected = tensor_a + tensor_b
    actual = pytree.tree_add(tensor_a, tensor_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] + tree_b[0], tree_a[1] + tree_b[1])
    actual = pytree.tree_add(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_add_scalar_mul() -> None:
    expected = (tree_a[0] + tree_b[0], tree_a[1] + tree_b[1])
    actual = pytree.tree_add_scalar_mul(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] + 0.5 * tree_b[0], tree_a[1] + 0.5 * tree_b[1])
    actual = pytree.tree_add_scalar_mul(tree_a, tree_b, 0.5)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_sub() -> None:
    expected = tensor_a - tensor_b
    actual = pytree.tree_sub(tensor_a, tensor_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] - tree_b[0], tree_a[1] - tree_b[1])
    actual = pytree.tree_sub(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_sub_scalar_mul() -> None:
    expected = (tree_a[0] - tree_b[0], tree_a[1] - tree_b[1])
    actual = pytree.tree_sub_scalar_mul(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] - 0.5 * tree_b[0], tree_a[1] - 0.5 * tree_b[1])
    actual = pytree.tree_sub_scalar_mul(tree_a, tree_b, 0.5)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_mul() -> None:
    expected = tensor_a * tensor_b
    actual = pytree.tree_mul(tensor_a, tensor_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] * tree_b[0], tree_a[1] * tree_b[1])
    actual = pytree.tree_mul(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_matmul() -> None:
    tree_a = (torch.randn(20, 10), torch.randn(20, 1))
    tree_b = (torch.randn(10, 20), torch.randn(1, 20))
    tensor_a = torch.randn(10, 20)
    tensor_b = torch.randn(20)
    expected = tensor_a @ tensor_b
    actual = pytree.tree_matmul(tensor_a, tensor_b)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (tree_a[0] @ tree_b[0], tree_a[1] @ tree_b[1])
    actual = pytree.tree_matmul(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_scalar_mul() -> None:
    expected = 0.5 * tensor_a
    actual = pytree.tree_scalar_mul(0.5, tensor_a)
    helpers.assert_pytree_all_close(actual, expected)

    expected = (0.5 * tree_a[0], 0.5 * tree_a[1])
    actual = pytree.tree_scalar_mul(0.5, tree_a)
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_truediv() -> None:
    expected = (tree_a[0] / tree_b[0], tree_a[1] / tree_b[1])
    actual = pytree.tree_truediv(tree_a, tree_b)
    helpers.assert_pytree_all_close(actual, expected)

    actual = pytree.tree_truediv(tree_a_dict, tree_b_dict)
    expected = (
        torch.tensor(1.0),
        {'k1': torch.tensor(0.5), 'k2': (torch.tensor(1.0 / 3.0), torch.tensor(0.25))},
        torch.tensor(0.2),
    )
    helpers.assert_pytree_all_close(actual, expected)


def test_tree_vdot_real() -> None:
    expected = torch.vdot(tensor_a, tensor_b).real
    actual = torch.tensor(pytree.tree_vdot_real(tensor_a, tensor_b))
    helpers.assert_pytree_all_close(actual, expected)

    expected = (
        torch.vdot(tree_a[0].contiguous().view(-1), tree_b[0].contiguous().view(-1))
        + torch.vdot(tree_a[1].contiguous().view(-1), tree_b[1].contiguous().view(-1))
    ).real
    actual = torch.tensor(pytree.tree_vdot_real(tree_a, tree_b))
    helpers.assert_all_close(actual, expected)

    tensor_a_complex = torch.randn(20, dtype=torch.cfloat)
    tensor_b_complex = torch.randn(20, dtype=torch.cfloat)
    expected = torch.vdot(tensor_a_complex, tensor_b_complex).real
    actual = torch.tensor(pytree.tree_vdot_real(tensor_a_complex, tensor_b_complex))
    helpers.assert_pytree_all_close(actual, expected)

    tree_a_complex, tree_b_complex = pytree.tree_map(
        lambda x: torch.randn(x.size(), dtype=torch.cfloat),
        (tree_a, tree_b),
    )
    expected = (
        torch.vdot(tree_a_complex[0].contiguous().view(-1), tree_b_complex[0].contiguous().view(-1))
        + torch.vdot(
            tree_a_complex[1].contiguous().view(-1),
            tree_b_complex[1].contiguous().view(-1),
        )
    ).real
    actual = torch.tensor(pytree.tree_vdot_real(tree_a_complex, tree_b_complex))
    helpers.assert_all_close(actual, expected)


@helpers.parametrize(
    tree_name=[
        'tree_a',
        'tree_b',
        'tree_a_dict',
        'tree_b_dict',
        'tensor_a',
        'tensor_b',
    ],
)
def test_tree_wait(tree_name: str) -> None:
    tree = globals()[tree_name]

    future_tree = pytree.tree_map(lambda x: torch.futures.Future(), tree)
    new_future_tree = pytree.tree_map(
        lambda fut: fut.then(lambda f: torch.square(f.wait()) + 1.0),
        future_tree,
    )
    pytree.tree_map_(lambda fut, x: fut.set_result(x), future_tree, tree)

    expected = pytree.tree_map(lambda x: torch.square(x) + 1.0, tree)
    actual = pytree.tree_wait(new_future_tree)
    assert all(fut.done() for fut in pytree.tree_leaves(new_future_tree))
    helpers.assert_pytree_all_close(actual, expected)
