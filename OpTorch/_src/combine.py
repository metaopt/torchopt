from . import base


def chain(
        *args: base.GradientTransformation
) -> base.GradientTransformation:
    """Applies a list of chainable update transformations.

  Given a sequence of chainable transforms, `chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A single (init_fn, update_fn) tuple.
  """

    init_fns, update_fns = zip(*args)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, inplace=True):
        if len(update_fns) != len(state):
            raise ValueError('The number of updates and states has to be the same in '
                             'chain! Make sure you have called init first!')
        updates = list(updates)
        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, inplace)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return base.GradientTransformation(init_fn, update_fn)
