Basics
======

This section describes useful concepts across TorchOpt.

TorchOpt Types
--------------

.. currentmodule:: torchopt.base

.. autoclass:: GradientTransformation
    :members:

.. autoclass:: TransformInitFn
    :members:

.. autoclass:: TransformUpdateFn
    :members:

.. autoclass:: OptState
    :members:

.. autoclass:: Params
    :members:

.. autoclass:: Updates
    :members:

PyTrees
-------

`PyTrees <https://github.com/metaopt/optree#pytrees>`_ are an essential concept in TorchOpt.
They can be thought as a generalization of vectors.
They are a way to structure parameters or weights using tuples and dictionaries.
Many solvers in TorchOpt have native support for pytrees.

Floating-Point Precision
------------------------

TorchOpt uses single (32-bit) floating precision (``torch.float32``) by default.
However, for some algorithms, this may not be enough.
Double (64-bit) floating precision (``torch.float64``) can be enabled by adding the following lines at the beginning of the file:

.. code-block:: python

    import torch

    torch.set_default_dtype(torch.float64)
