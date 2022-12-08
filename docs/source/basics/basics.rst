Basics
======

This section describes useful concepts across TorchOpt.

TorchOpt Types
--------------

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

Pytrees
-------

`Pytrees <https://optree.readthedocs.io/en/latest/>`_ are an essential
concept in torchopt. They can be thought as a generalization of vectors.
They are a way to structure parameters or weights using tuples and
dictionaries. Many solvers in torchopt have native support for pytrees.

Half precision
----------------

torchopt uses single (32-bit) floating precision by default. However, for some
algorithms, this may not be enough. Double (64-bit) floating precision can be
enabled by adding the following at the beginning of the file::
