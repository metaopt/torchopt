.. _implicit_diff:

Implicit Gradient differentiation
=================================

Argmin differentiation
----------------------

Argmin differentiation is the task of differentiating a minimization problem's solution with respect to its inputs.
Namely, given

.. math::

    x^\star(\theta) := \underset{x}{\text{argmin}} f(x, \theta),

we would like to compute the Jacobian :math:`\partial x^\star(\theta)`.
This is usually done either by implicit differentiation or by autodiff through an algorithm's unrolled iterates.

Custom solvers
--------------

.. autosummary::
    :toctree: ../_autosummary

    torchopt.diff.implicit.custom_root

TorchOpt provides the ``custom_root`` decorators, for easily adding implicit differentiation on top of any existing solver.

.. topic:: Examples

    * :ref:`implicit_diff.py`