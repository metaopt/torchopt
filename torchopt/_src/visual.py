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
# This file is modified from:
# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
# ==============================================================================

import warnings
from collections import namedtuple
from typing import Dict, Generator

import torch
from graphviz import Digraph
from pkg_resources import parse_version


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = '_saved_'


def get_fn_name(fn, show_attrs, max_attr_chars):
    """Returns function name."""
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = {}
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX) :]
        if torch.is_tensor(val):
            attrs[attr] = '[saved tensor]'
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
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

    def truncate(s):  # pylint: disable=invalid-name
        return s[: col2width - 3] + '...' if len(s) > col2width else s

    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


# mypy: ignore-errors
# pylint: disable-next=too-many-branches,too-many-statements,too-many-locals
def make_dot(
    var: torch.Tensor, params=None, show_attrs=False, show_saved=False, max_attr_chars=50
) -> Digraph:
    """Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node represents a tensor
    and is either blue, orange, or green:

        - **Blue**
            Reachable leaf tensors that requires grad (tensors whose :attr:`grad` fields will be
            populated during :meth:`backward`).
        - **Orange**
            Saved tensors of custom autograd functions as well as those saved by built-in backward
            nodes.
        - **Green**
            Tensor passed in as outputs.
        - **Dark green**
            If any output is a view, we represent its base tensor with a dark green node.

    Args:
        var: Output tensor.
        params: ([dict of (name, tensor) or state_dict])
            Parameters to add names to node that requires grad.
        show_attrs: Whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: Whether to display saved tensor nodes that are not by custom autograd
            functions. Saved tensor nodes for custom functions, if present, are always displayed.
            (Requires PyTorch version >= 1.9)
        max_attr_chars: If ``show_attrs`` is :data:`True`, sets max number of characters to display
            for any given attribute.
    """
    if parse_version(torch.__version__) < parse_version('1.9') and (show_attrs or show_saved):
        warnings.warn(
            'make_dot: showing grad_fn attributes and saved variables '
            'requires PyTorch version >= 1.9. (This does NOT apply to '
            'saved tensors saved by custom autograd functions.)'
        )

    param_map = {}

    if params is not None:
        from torchopt._src.utils import _ModuleState  # pylint: disable=import-outside-toplevel

        if isinstance(params, _ModuleState):
            param_map.update(params.visual_contents)
        elif isinstance(params, Dict):
            param_map.update({v: k for k, v in params.items()})
        elif isinstance(params, Generator):
            param_map.update({v: k for k, v in params})
        else:
            for param in params:
                if isinstance(param, _ModuleState):
                    param_map.update(param.visual_contents)
                elif isinstance(param, Generator):
                    param_map.update({v: k for k, v in param})
                else:
                    param_map.update({v: k for k, v in param.items()})

    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='10',
        ranksep='0.1',
        height='0.2',
        fontname='monospace',
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size='12,12'))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(map(str, size)) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[var] if var in param_map else ''
        return f'{name}\n{size_to_str(var.size())}'

    def get_var_name_with_flag(var):
        if var in param_map:
            return f'{param_map[var][0]}\n{size_to_str(param_map[var][1].size())}'
        return None

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
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
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir='none')
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
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

    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if var.grad_fn:
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        # pylint: disable=protected-access
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style='dotted')

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:  # pylint: disable=invalid-name
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.5, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + ',' + str(size)
    dot.graph_attr.update(size=size_str)
