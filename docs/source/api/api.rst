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

Optimizer Hooks
===============

.. currentmodule:: torchopt._src.hook

.. autosummary::

    register_hook
    zero_nan_hook

Hook
~~~~

.. autofunction:: register_hook
.. autofunction:: zero_nan_hook

Gradient Transformation
=======================

.. currentmodule:: torchopt._src.clip

.. autosummary::

    clip_grad_norm

Transforms
~~~~~~~~~~

.. autofunction:: clip_grad_norm

Optimizer Schedules
===================

.. currentmodule:: torchopt._src.schedule

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

.. currentmodule:: torchopt._src.combine

.. autosummary::

    chain

Chain
~~~~~

.. autofunction:: chain


Implicit Gradient
=================

.. currentmodule:: torchopt._src.implicit_diff

.. autosummary::

    custom_root

Root Solver
~~~~~~~~~~~~~~~~~~

.. autofunction:: custom_root


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

.. currentmodule:: torchopt._src.visual

.. autosummary::

    make_dot

Make Dot
~~~~~~~~

.. autofunction:: make_dot
