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


Common Optimizers
=================

.. currentmodule:: torchopt

.. autosummary::

    adam
    sgd
    rmsprop
    MetaAdam
    MetaSGD
    MetaRMSProp


Adam
~~~~

.. autofunction:: adam

SGD
~~~

.. autofunction:: sgd

RMSProp
~~~~~~~

.. autofunction:: rmsprop

MetaAdam
~~~~~~~~

.. autofunction:: MetaAdam

MetaSGD
~~~~~~~

.. autofunction:: MetaSGD

MetaRMSProp
~~~~~~~~~~~

.. autofunction:: MetaRMSProp


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
