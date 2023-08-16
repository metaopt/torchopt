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
# This file is modified from:
# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
# ==============================================================================
"""Computation graph visualization."""

from __future__ import annotations

from typing import Any, Generator, Iterable, Mapping, cast

import torch
from graphviz import Digraph

from torchopt import pytree
from torchopt.typing import TensorTree
from torchopt.utils import ModuleState


__all__ = ['make_dot', 'resize_graph']


# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = '_saved_'


def get_fn_name(fn: Any, show_attrs: bool, max_attr_chars: int) -> str:
    """Return function name."""
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = {}
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX) :]
        if isinstance(val, torch.Tensor):
            attrs[attr] = '[saved tensor]'
        elif isinstance(val, tuple) and any(isinstance(t, torch.Tensor) for t in val):
            attrs[attr] = '[saved tensors]'
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(map(len, attrs))
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = '-' * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width) + 's'

    def truncate(s: str) -> str:  # pylint: disable=invalid-name
        return s[: col2width - 3] + '...' if len(s) > col2width else s

    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


# pylint: disable-next=too-many-branches,too-many-statements,too-many-locals
def make_dot(
    var: TensorTree,
    params: (
        Mapping[str, torch.Tensor]
        | ModuleState
        | Generator
        | Iterable[Mapping[str, torch.Tensor] | ModuleState | Generator]
        | None
    ) = None,
    show_attrs: bool = False,
    show_saved: bool = False,
    max_attr_chars: int = 50,
) -> Digraph:
    """Produce Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node represents a tensor
    and is either blue, orange, or green:

        - **Blue**
            Reachable leaf tensors that requires grad (tensors whose ``grad`` fields will be
            populated during :meth:`backward`).
        - **Orange**
            Saved tensors of custom autograd functions as well as those saved by built-in backward
            nodes.
        - **Green**
            Tensor passed in as outputs.
        - **Dark green**
            If any output is a view, we represent its base tensor with a dark green node.

    Args:
        var (Tensor or sequence of Tensor): Output tensor.
        params: (dict[str, Tensor], ModuleState, iterable of tuple[str, Tensor], or None, optional):
            Parameters to add names to node that requires grad. (default: :data:`None`)
        show_attrs (bool, optional): Whether to display non-tensor attributes of backward nodes.
            (default: :data:`False`)
        show_saved (bool, optional): Whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if present, are always
            displayed. (default: :data:`False`)
        max_attr_chars (int, optional): If ``show_attrs`` is :data:`True`, sets max number of
            characters to display for any given attribute. (default: :const:`50`)
    """
    param_map = {}

    if params is not None:
        if isinstance(params, ModuleState) and params.visual_contents is not None:
            param_map.update(params.visual_contents)
        elif isinstance(params, Mapping):
            param_map.update({v: k for k, v in params.items()})
        elif isinstance(params, Generator):
            param_map.update({v: k for k, v in params})
        else:
            for param in params:
                if isinstance(param, ModuleState) and param.visual_contents is not None:
                    param_map.update(param.visual_contents)
                elif isinstance(param, Generator):
                    param_map.update({v: k for k, v in param})
                else:
                    param_map.update({v: k for k, v in cast(Mapping, param).items()})

    node_attr = {
        'style': 'filled',
        'shape': 'box',
        'align': 'left',
        'fontsize': '10',
        'ranksep': '0.1',
        'height': '0.2',
        'fontname': 'monospace',
    }
    dot = Digraph(node_attr=node_attr, graph_attr={'size': '12,12'})
    seen = set()

    def size_to_str(size: tuple[int, ...]) -> str:
        return '(' + (', ').join(map(str, size)) + ')'

    def get_var_name(var: torch.Tensor, name: str | None = None) -> str:
        if not name:
            name = param_map[var] if var in param_map else ''
        return f'{name}\n{size_to_str(var.size())}'

    def get_var_name_with_flag(var: torch.Tensor) -> str | None:
        if var in param_map:
            return f'{param_map[var][0]}\n{size_to_str(param_map[var][1].size())}'
        return None

    def add_nodes(fn: Any) -> None:  # pylint: disable=too-many-branches
        assert not isinstance(fn, torch.Tensor)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX) :]
                if isinstance(val, torch.Tensor):
                    dot.edge(str(id(fn)), str(id(val)), dir='none')
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if isinstance(t, torch.Tensor):
                            name = f'{attr}[{i}]'
                            dot.edge(str(id(fn)), str(id(t)), dir='none')
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        fn_name = get_fn_name(fn, show_attrs, max_attr_chars)
        fn_fillcolor = None
        var_name = get_var_name_with_flag(fn)
        if var_name is not None:
            fn_name = f'{fn_name}\n{var_name}'
            fn_fillcolor = 'lightblue'

        # add the node for this grad_fn
        dot.node(str(id(fn)), fn_name, fillcolor=fn_fillcolor)

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                dot.edge(str(id(t)), str(id(fn)))
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')

    def add_base_tensor(
        v: torch.Tensor,  # pylint: disable=invalid-name
        color: str = 'darkolivegreen1',
    ) -> None:
        if v in seen:
            return
        seen.add(v)
        dot.node(str(id(v)), get_var_name(v), fillcolor=color)
        if v.grad_fn:
            add_nodes(v.grad_fn)
            dot.edge(str(id(v.grad_fn)), str(id(v)))
        # pylint: disable=protected-access
        if v._is_view():
            add_base_tensor(v._base, color='darkolivegreen3')  # type: ignore[arg-type]
            dot.edge(str(id(v._base)), str(id(v)), style='dotted')

    # handle multiple outputs
    pytree.tree_map_(add_base_tensor, var)

    resize_graph(dot)

    return dot


def resize_graph(dot: Digraph, size_per_element: float = 0.5, min_size: float = 12.0) -> None:
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + ',' + str(size)
    dot.graph_attr.update(size=size_str)
