TorchOpt Optimizer
==================

.. currentmodule:: torchopt

.. autosummary::

    Optimizer
    MetaOptimizer

Optimizer
~~~~~~~~~

.. autoclass:: Optimizer
    :members:

MetaOptimizer
~~~~~~~~~~~~~

.. autoclass:: MetaOptimizer
    :members:

------

Functional Optimizers
=====================

.. currentmodule:: torchopt

.. autosummary::

    FuncOptimizer
    adam
    sgd
    rmsprop
    adamw

Wrapper for Function Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FuncOptimizer
    :members:

Functional Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adam

Functional AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adamw

Functional SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sgd

Functional RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rmsprop

------

Classic Optimizers
==================

.. currentmodule:: torchopt

.. autosummary::

    Adam
    SGD
    RMSProp
    AdamW

Classic Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Adam

Classic AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdamW

Classic SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SGD

Classic RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RMSProp

------

Differentiable Meta-Optimizers
==============================

.. currentmodule:: torchopt

.. autosummary::

    MetaAdam
    MetaSGD
    MetaRMSProp
    MetaAdamW

Differentiable Meta-Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdam

Differentiable Meta-AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdamW

Differentiable Meta-SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaSGD

Differentiable Meta-RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaRMSProp

------

Implicit Differentiation
========================

.. currentmodule:: torchopt.diff.implicit

.. autosummary::

    custom_root
    nn.ImplicitMetaGradientModule

Custom Solvers
~~~~~~~~~~~~~~

.. autofunction:: custom_root


Implicit Meta-Gradient Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchopt.diff.implicit.nn

.. autoclass:: ImplicitMetaGradientModule
    :members:

------

Linear System Solvers
=====================

.. currentmodule:: torchopt.linear_solve

.. autosummary::

    solve_cg
    solve_normal_cg
    solve_inv

Indirect Solvers
~~~~~~~~~~~~~~~~

.. autofunction:: solve_cg
.. autofunction:: solve_normal_cg
.. autofunction:: solve_inv

------

Zero-Order Differentiation
==========================

.. currentmodule:: torchopt.diff.zero_order

.. autosummary::

    zero_order
    nn.ZeroOrderGradientModule

Decorators
~~~~~~~~~~

.. autofunction:: zero_order


Zero-order Gradient Module
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchopt.diff.zero_order.nn

.. autoclass:: ZeroOrderGradientModule
    :members:

------

Optimizer Hooks
===============

.. currentmodule:: torchopt.hook

.. autosummary::

    register_hook
    zero_nan_hook
    nan_to_num_hook

Hook
~~~~

.. autofunction:: register_hook
.. autofunction:: zero_nan_hook
.. autofunction:: nan_to_num_hook

------

Gradient Transformation
=======================

.. currentmodule:: torchopt

.. autosummary::

    clip_grad_norm
    nan_to_num

Transforms
~~~~~~~~~~

.. autofunction:: clip_grad_norm
.. autofunction:: nan_to_num

Optimizer Schedules
===================

.. currentmodule:: torchopt.schedule

.. autosummary::

    linear_schedule
    polynomial_schedule

Schedules
~~~~~~~~~

.. autofunction:: linear_schedule
.. autofunction:: polynomial_schedule


Apply Parameter Updates
=======================

.. currentmodule:: torchopt

.. autosummary::

    apply_updates

Apply Updates
~~~~~~~~~~~~~

.. autofunction:: apply_updates

Combining Optimizers
====================

.. currentmodule:: torchopt.combine

.. autosummary::

    chain

Chain
~~~~~

.. autofunction:: chain


General Utilities
=================

.. currentmodule:: torchopt

.. autosummary::

    extract_state_dict
    recover_state_dict
    stop_gradient

Extract State Dict
~~~~~~~~~~~~~~~~~~

.. autofunction:: extract_state_dict

Recover State Dict
~~~~~~~~~~~~~~~~~~

.. autofunction:: recover_state_dict

Stop Gradient
~~~~~~~~~~~~~

.. autofunction:: stop_gradient


Visualizing Gradient Flow
=========================

.. currentmodule:: torchopt.visual

.. autosummary::

    make_dot

Make Dot
~~~~~~~~

.. autofunction:: make_dot
