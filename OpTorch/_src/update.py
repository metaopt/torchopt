import jax

from . import base


def apply_updates(params: base.Params, updates: base.Updates, inplace: bool = True) -> base.Params:
    """Applies an update to the corresponding parameters.

  This is a utility functions that applies an update to a set of parameters, and
  then returns the updated parameters to the caller. As an example, the update
  may be a gradient transformed by a sequence of`GradientTransformations`. This
  function is exposed for convenience, but it just adds updates and parameters;
  you may also apply updates to parameters manually, using `tree_map`
  (e.g. if you want to manipulate updates in custom ways before applying them).

  Args:
    params: a tree of parameters.
    updates: a tree of updates, the tree structure and the shape of the leaf
    nodes must match that of `params`.
    inplace: if True, will update params in a inplace manner.

  Returns:
    Updated parameters, with same structure, shape and type as `params`.
  """
    if inplace:
        def f(p, u):
            if u is not None:
                p.data.add_(u)
            return p
    else:
        def f(p, u):
            return p.add(u) if u is not None else p
    return jax.tree_map(f, list(params), list(updates))
