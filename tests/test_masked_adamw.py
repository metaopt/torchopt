# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import torchopt
from torchopt import pytree
from torchopt.alias.utils import _get_use_chain_flat, _set_use_chain_flat


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(params=[True, False], ids=['flat', 'pytree'])
def use_chain_flat(request: pytest.FixtureRequest) -> Generator[bool, None, None]:
    previous = _get_use_chain_flat()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    _set_use_chain_flat(request.param)
    try:
        yield request.param
    finally:
        _set_use_chain_flat(previous)
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('mask_values', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('callable_mask', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('use_accelerated_op', [True, False])
def test_masked_adamw_matches_parameter_groups(
    use_chain_flat: bool,
    mask_values: tuple[bool, bool],
    callable_mask: bool,
    inplace: bool,
    use_accelerated_op: bool,
) -> None:
    params = [
        torch.tensor([[0.7, -1.2]], dtype=torch.float64),
        torch.tensor([0.4, -0.6], dtype=torch.float64),
        torch.tensor([2.3], dtype=torch.float64),
    ]
    reference = [param.clone() for param in params]
    mask = [mask_values[0], mask_values[1], mask_values[1]]
    if callable_mask:

        def mask_fn(tree):
            return pytree.tree_map(lambda p: mask_values[0 if p.ndim == 2 else 1], tree)

        mask = mask_fn
    optimizer = torchopt.adamw(
        lr=0.2,
        betas=(0.8, 0.9),
        weight_decay=0.1,
        mask=mask,
        use_accelerated_op=use_accelerated_op,
    )
    state = optimizer.init(params)
    reference_optimizer = torch.optim.AdamW(
        [
            {'params': [reference[0]], 'weight_decay': 0.1 if mask_values[0] else 0.0},
            {'params': reference[1:], 'weight_decay': 0.1 if mask_values[1] else 0.0},
        ],
        lr=0.2,
        betas=(0.8, 0.9),
    )
    for step in range(4):
        grads = [2 * param + step / 10 for param in params[:2]] + [None]
        for param, grad in zip(reference, grads):
            param.grad = None if grad is None else grad.clone()
        updates, state = optimizer.update(grads, state, params=params, inplace=inplace)
        assert updates[2] is None
        params = torchopt.apply_updates(params, updates, inplace=inplace)
        reference_optimizer.step()
        for actual, expected in zip(params, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_masked_adamw_meta_gradient(use_chain_flat: bool) -> None:
    initial = [
        torch.tensor([[0.7, -1.2]], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.4, -0.6], dtype=torch.float64, requires_grad=True),
    ]
    reference_initial = [param.detach().clone().requires_grad_() for param in initial]
    params = initial
    reference = reference_initial
    optimizer = torchopt.adamw(
        lr=0.2,
        betas=(0.8, 0.9),
        eps=1e-4,
        eps_root=1e-3,
        weight_decay=0.1,
        mask=lambda tree: pytree.tree_map(lambda p: p.ndim != 1, tree),
    )
    state = optimizer.init(params)
    first_moments = [torch.zeros_like(param) for param in reference]
    second_moments = [torch.zeros_like(param) for param in reference]
    for step in range(1, 4):
        loss = (params[0] + params[1]).square().sum()
        grads = list(torch.autograd.grad(loss, params, create_graph=True))
        updates, state = optimizer.update(grads, state, params=params, inplace=False)
        params = torchopt.apply_updates(params, updates, inplace=False)

        reference_loss = (reference[0] + reference[1]).square().sum()
        reference_grads = torch.autograd.grad(reference_loss, reference, create_graph=True)
        next_reference = []
        for index, (param, grad) in enumerate(zip(reference, reference_grads)):
            first_moments[index] = 0.8 * first_moments[index] + 0.2 * grad
            second_moments[index] = 0.9 * second_moments[index] + 0.1 * grad.square()
            mean = first_moments[index] / (1 - 0.8**step)
            variance = second_moments[index] / (1 - 0.9**step)
            update = mean / ((variance + 1e-3).sqrt() + 1e-4)
            if index == 0:
                update = update + 0.1 * param
            next_reference.append(param - 0.2 * update)
        reference = next_reference
        for actual, expected in zip(params, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    outer_loss = sum(param.square().sum() for param in params)
    reference_outer_loss = sum(param.square().sum() for param in reference)
    meta_grads = torch.autograd.grad(outer_loss, initial)
    reference_meta_grads = torch.autograd.grad(reference_outer_loss, reference_initial)
    for actual, expected in zip(meta_grads, reference_meta_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('inplace', [True, False])
def test_masked_adam_state(use_chain_flat: bool, inplace: bool) -> None:
    params = [torch.tensor([2.0]), torch.tensor([3.0])]
    inner = torchopt.transform.scale_by_adam.impl(already_flattened=use_chain_flat)
    optimizer = torchopt.transform.masked.impl(
        inner,
        [True, False],
        already_flattened=use_chain_flat,
    )
    state = optimizer.init(params)
    # Adam stores two moments and a counter for only the selected parameter.
    assert len(pytree.tree_leaves(state)) == 3
    for _ in range(3):
        grads = [torch.tensor([2.0]), torch.tensor([3.0])]
        passed_through = grads[1]
        updates, state = optimizer.update(grads, state, inplace=inplace)
        assert updates[1] is passed_through
        torch.testing.assert_close(updates[0], torch.tensor([2.0 / (2.0 + 1e-8)]))
        assert len(pytree.tree_leaves(state)) == 3


@pytest.mark.parametrize('callable_mask', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('mask_kind', ['full', 'prefix', 'all', 'none'])
def test_adamw_mask_preserves_parameter_tree(
    use_chain_flat: bool,
    callable_mask: bool,
    inplace: bool,
    mask_kind: str,
) -> None:
    params = {
        'layer': {
            'weight': torch.tensor([[2.0, 3.0]]),
            'bias': torch.tensor([4.0, 5.0]),
        },
        'head': [torch.tensor([6.0]), torch.tensor([7.0])],
    }
    if mask_kind == 'full':
        mask = {'layer': {'weight': True, 'bias': False}, 'head': [False, False]}
    elif mask_kind == 'prefix':
        mask = {'layer': {'weight': True, 'bias': False}, 'head': False}
    else:
        mask = mask_kind == 'all'
    if callable_mask:
        original_mask = mask

        def mask_fn(tree):
            # A callable must see the original tree, including the nested list.
            assert tree['layer']['weight'].ndim == 2
            assert len(tree['head']) == 2
            return original_mask

        mask = mask_fn
    expected = pytree.tree_map(torch.clone, params)
    if mask_kind == 'all':
        expected = pytree.tree_map(lambda p: p * (1 - 0.2 * 0.1), expected)
    elif mask_kind != 'none':
        expected['layer']['weight'] *= 1 - 0.2 * 0.1
    optimizer = torchopt.adamw(lr=0.2, weight_decay=0.1, mask=mask)
    state = optimizer.init(params)
    updates, _ = optimizer.update(
        pytree.tree_map(torch.zeros_like, params),
        state,
        params=params,
        inplace=inplace,
    )
    result = torchopt.apply_updates(params, updates, inplace=inplace)
    assert pytree.tree_structure(result) == pytree.tree_structure(expected)
    for actual, target in zip(pytree.tree_leaves(result), pytree.tree_leaves(expected)):
        torch.testing.assert_close(actual, target)


@pytest.mark.parametrize('inplace', [True, False])
def test_prefix_mask_excludes_optimizer_state(inplace: bool) -> None:
    params = {'active': [torch.tensor([2.0])], 'frozen': {'bias': torch.tensor([3.0])}}
    optimizer = torchopt.transform.masked(
        torchopt.transform.scale_by_adam(),
        {'active': True, 'frozen': False},
    )
    state = optimizer.init(params)
    assert len(pytree.tree_leaves(state)) == 3
    updates, state = optimizer.update(params, state, inplace=inplace)
    assert updates['frozen'] is params['frozen']
    assert len(pytree.tree_leaves(state)) == 3
    torch.testing.assert_close(updates['active'][0], torch.tensor([2.0 / (2.0 + 1e-8)]))


@pytest.mark.filterwarnings(
    'ignore:There is a performance drop because we have not yet implemented the batching rule '
    'for aten..addcmul_:UserWarning',
)
def test_masked_adamw_vmap(use_chain_flat: bool) -> None:
    def step(params, inputs):
        optimizer = torchopt.adamw(
            lr=0.2,
            weight_decay=0.1,
            mask=lambda tree: pytree.tree_map(lambda p: p.ndim != 1, tree),
        )
        state = optimizer.init(params)
        grads = torch.func.grad(
            lambda p: (p['weight'] @ inputs + p['bias']).square().sum(),
        )(params)
        updates, _ = optimizer.update(grads, state, params=params, inplace=False)
        return torchopt.apply_updates(params, updates, inplace=False)

    params = {
        'weight': torch.tensor([[[0.7, -1.2]], [[1.3, 0.1]]]),
        'bias': torch.tensor([[0.4], [-0.6]]),
    }
    inputs = torch.tensor([[0.5, 0.8], [-1.2, 0.6]])
    actual = torch.vmap(step)(params, inputs)
    reference = [
        step(pytree.tree_map(lambda p, index=index: p[index], params), inputs[index])
        for index in range(2)
    ]
    expected = pytree.tree_map(lambda *values: torch.stack(values), *reference)
    for result, target in zip(pytree.tree_leaves(actual), pytree.tree_leaves(expected)):
        torch.testing.assert_close(result, target)


@pytest.mark.parametrize('inplace', [True, False])
def test_callable_mask_receives_updates(use_chain_flat: bool, inplace: bool) -> None:
    params = [torch.tensor([2.0]), torch.tensor([-3.0]), torch.tensor([4.0])]
    grads = [torch.tensor([-0.5]), torch.tensor([0.25]), None]
    mask_inputs = []

    def mask_fn(tree):
        mask_inputs.append(tree)
        return pytree.tree_map(lambda value: bool((value > 0).all()), tree)

    optimizer = torchopt.transform.add_decayed_weights.impl(
        weight_decay=0.1,
        mask=mask_fn,
        already_flattened=use_chain_flat,
    )
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params=params, inplace=inplace)
    assert mask_inputs[0] is params
    assert mask_inputs[1] is grads
    torch.testing.assert_close(updates[0], torch.tensor([-0.5]))
    torch.testing.assert_close(updates[1], torch.tensor([-0.05]))
    assert updates[2] is None


@pytest.mark.parametrize('inplace', [True, False])
def test_masked_adam_state_with_missing_updates(use_chain_flat: bool, inplace: bool) -> None:
    params = [torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0])]
    inner = torchopt.transform.scale_by_adam.impl(already_flattened=use_chain_flat)
    optimizer = torchopt.transform.masked.impl(
        inner,
        [True, False, True],
        already_flattened=use_chain_flat,
    )
    state = optimizer.init(params)
    for _ in range(3):
        updates, state = optimizer.update(
            [torch.tensor([2.0]), None, None],
            state,
            inplace=inplace,
        )
        assert updates[1] is None
        assert updates[2] is None
        torch.testing.assert_close(state.inner_state.mu[2], torch.tensor([0.0]))
        torch.testing.assert_close(state.inner_state.nu[2], torch.tensor([0.0]))
        assert state.inner_state.count[2].item() == 0
    updates, state = optimizer.update(
        [torch.tensor([2.0]), None, torch.tensor([4.0])],
        state,
        inplace=inplace,
    )
    torch.testing.assert_close(updates[2], torch.tensor([4.0 / (4.0 + 1e-8)]))
    assert state.inner_state.count[2].item() == 1
