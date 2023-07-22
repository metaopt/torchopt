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
    adadelta
    adagrad
    adam
    adamw
    adamax
    radam
    rmsprop
    sgd

Wrapper for Function Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FuncOptimizer
    :members:

Functional AdaDelta Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adadelta

Functional AdaGrad Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adagrad

Functional Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adam

Functional AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adamw

Functional AdaMax Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adamax

Functional RAdam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: radam

Functional RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rmsprop

Functional SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sgd

------

Classic Optimizers
==================

.. currentmodule:: torchopt

.. autosummary::

    AdaDelta
    Adadelta
    AdaGrad
    Adagrad
    Adam
    AdamW
    AdaMax
    Adamax
    RAdam
    RMSProp
    SGD

Classic AdaDelta Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaDelta

Classic AdaGrad Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaGrad

Classic Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Adam

Classic AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdamW

Classic AdaMax Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaMax

Classic RAdam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RAdam

Classic RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RMSProp

Classic SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SGD

------

Differentiable Meta-Optimizers
==============================

.. currentmodule:: torchopt

.. autosummary::

    MetaAdaDelta
    MetaAdadelta
    MetaAdaGrad
    MetaAdagrad
    MetaAdam
    MetaAdamW
    MetaAdaMax
    MetaAdamax
    MetaRAdam
    MetaRMSProp
    MetaSGD

Differentiable Meta-AdaDelta Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdaDelta

Differentiable Meta-AdaGrad Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdaGrad

Differentiable Meta-Adam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdam

Differentiable Meta-AdamW Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdamW

Differentiable Meta-AdaMax Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaAdaMax

Differentiable Meta-RAdam Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaRAdam

Differentiable Meta-RMSProp Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaRMSProp

Differentiable Meta-SGD Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MetaSGD

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


Distributed Utilities
=====================

.. currentmodule:: torchopt.distributed

Initialization and Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    auto_init_rpc
    barrier

.. autofunction:: auto_init_rpc
.. autofunction:: barrier

Process group information
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    get_world_info
    get_world_rank
    get_rank
    get_world_size
    get_local_rank
    get_local_world_size
    get_worker_id

.. autofunction:: get_world_info
.. autofunction:: get_world_rank
.. autofunction:: get_rank
.. autofunction:: get_world_size
.. autofunction:: get_local_rank
.. autofunction:: get_local_world_size
.. autofunction:: get_worker_id

Worker selection
~~~~~~~~~~~~~~~~

.. autosummary::

    on_rank
    not_on_rank
    rank_zero_only
    rank_non_zero_only

.. autofunction:: on_rank
.. autofunction:: not_on_rank
.. autofunction:: rank_zero_only
.. autofunction:: rank_non_zero_only

Remote Procedure Call (RPC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    remote_async_call
    remote_sync_call

.. autofunction:: remote_async_call
.. autofunction:: remote_sync_call

Predefined partitioners and reducers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    dim_partitioner
    batch_partitioner
    mean_reducer
    sum_reducer

.. autofunction:: dim_partitioner
.. autofunction:: batch_partitioner
.. autofunction:: mean_reducer
.. autofunction:: sum_reducer

Function parallelization wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    parallelize
    parallelize_async
    parallelize_sync

.. autofunction:: parallelize
.. autofunction:: parallelize_async
.. autofunction:: parallelize_sync

Distributed Autograd
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchopt.distributed.autograd

.. autosummary::

    context
    get_gradients
    backward
    grad

.. autofunction:: context
.. autofunction:: get_gradients
.. autofunction:: backward
.. autofunction:: grad


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
