Implicit Gradient Differentiation
=================================

.. currentmodule:: torchopt.diff.implicit

Implicit Differentiation
------------------------

.. image:: /_static/images/implicit-gradient.png
    :width: 80%
    :align: center

Implicit differentiation is the task of differentiating the solution of a minimization problem with respect to its inputs.
Namely, given

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) \triangleq \underset{\boldsymbol{\theta}}{\mathop{\operatorname{argmin}}} ~ \mathcal{L}^{\text{in}} (\boldsymbol{\phi},\boldsymbol{\theta}).

By treating the solution :math:`\boldsymbol{\theta}^{\prime}` as an implicit function of :math:`\boldsymbol{\phi}`, the idea of implicit differentiation is to directly get analytical best-response derivatives :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})` by the implicit function theorem.
This is suitable for algorithms when the inner-level optimal solution is achieved :math:`\left. \frac{\partial \mathcal{L}^{\text{in}} (\boldsymbol{\phi}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \right\rvert_{\boldsymbol{\theta} = \boldsymbol{\theta}^{\prime}} = 0` (e.g., the function :math:`F` in the zfigure means the solution is obtained by unrolled gradient steps)


Differentiating a root. Let :math:`F: \mathbb{R}^d \times \mathbb{R}^n \rightarrow \mathbb{R}^d` be a user-provided mapping, capturing the optimality conditions of a problem. An optimal solution, denoted :math:`x^{\star}(\theta)`, should be a root of :math:`F` :

.. math::
    F\left(x^{\star}(\theta), \theta\right)=0 \text {. }

We can see :math:`x^{\star}(\theta)` as an implicitly defined function of :math:`\theta \in \mathbb{R}^n`, i.e., :math:`x^{\star}: \mathbb{R}^n \rightarrow \mathbb{R}^d`. More precisely, from the implicit function theorem [48, 57], we know that for :math:`\left(x_0, \theta_0\right)` satisfying :math:`F\left(x_0, \theta_0\right)=0` with a continuously differentiable :math:`F`, if the Jacobian :math:`\partial_1 F` evaluated at :math:`\left(x_0, \theta_0\right)` is a square invertible matrix, then there exists a function :math:`x^{\star}(\cdot)` defined on a neighborhood of :math:`\theta_0` such that :math:`x^{\star}(\theta_0)=x_0`. Furthermore, for all :math:`\theta` in this neighborhood, we have that :math:`F(x^{\star}(\theta), \theta)=0` and :math:`\partial x^{\star}(\theta)` exists. Using the chain rule, the Jacobian :math:`\partial x^{\star}(\theta)` satisfies

.. math::
    \partial_1 F(x^{\star}(\theta), \theta) \partial x^{\star}(\theta)+\partial_2 F(x^{\star}(\theta), \theta)=0 .

Computing :math:`\partial x^{\star}(\theta)` therefore boils down to the resolution of the linear system of equations

.. math::
    \underbrace{-\partial_1 F(x^{\star}(\theta), \theta)}_{A \in \mathbb{R}^{d \times d}} \underbrace{\partial x^{\star}(\theta)}_{J \in \mathbb{R}^{d \times n}}=\underbrace{\partial_2 F(x^{\star}(\theta), \theta)}_{B \in \mathbb{R}^{d \times n}} \text {. }

When (1) is a one-dimensional root finding problem :math:`(d=1)`, (2) becomes particularly simple since we then have :math:`\nabla x^{\star}(\theta)=B^{\top} / A`, where :math:`A` is a scalar value.


or reaches some stationary conditions :math:`F (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime}) = 0`, such as `IMAML <https://arxiv.org/abs/1909.04630>`_ and `DEQ <https://arxiv.org/abs/1909.01377>`_.

Custom Solvers
--------------

.. autosummary::

    torchopt.diff.implicit.custom_root

TorchOpt provides the :func:`custom_root` decorators, for easily adding implicit differentiation on top of any existing solver (also called forward optimization).
:func:`custom_root` requires users to define the stationary conditions for the problem solution (e.g., KKT conditions) and will automatically calculate the gradient for backward gradient computation.

Here is an example of the :func:`custom_root` decorators, which is also the **functional API** for implicit gradient.

.. code-block:: python

    # Functional API for implicit gradient
    def stationary(params, meta_params, data):
        # stationary condition construction
        return stationary condition

    # Decorator that wraps the function
    # Optionally specify the linear solver (conjugate gradient or Neumann series)
    @torchopt.diff.implicit.custom_root(stationary)
    def solve(params, meta_params, data):
        # Forward optimization process for params
        return optimal_params

    # Define params, meta_params and get data
    params, meta_prams, data = ..., ..., ...
    optimal_params = solve(params, meta_params, data)
    loss = outer_loss(optimal_params)

    meta_grads = torch.autograd.grad(loss, meta_params)

OOP API
~~~~~~~

.. autosummary::

    torchopt.nn.ImplicitMetaGradientModule

Coupled with PyTorch |torch.nn.Module|_, we also design the OOP API :class:`nn.ImplicitMetaGradientModule` for implicit gradient.
The core idea of :class:`nn.ImplicitMetaGradientModule` is to enable the gradient flow from ``self.parameters()`` (usually lower-level parameters) to ``self.meta_parameters()`` (usually the high-level parameters).
Users need to define the forward process ``forward()``, a stationary function ``optimality()`` (or ``objective()``), and inner-loop optimization ``solve``.

.. |torch.nn.Module| replace:: ``torch.nn.Module``
.. _torch.nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module

Here is an example of the OOP API.

.. code-block:: python

    from torchopt.nn import ImplicitMetaGradientModule

    # Inherited from the class ImplicitMetaGradientModule
    class InnerNet(ImplicitMetaGradientModule):
        def __init__(self, meta_module):
            ...

        def forward(self, batch):
            # Forward process
            ...

        def optimality(self, batch, labels):
            # Stationary condition construction for calculating implicit gradient
            # NOTE: If this method is not implemented, it will be automatically derived from the
            # gradient of the `objective` function.
            ...

        def objective(self, batch, labels):
            # Define the inner-loop optimization objective
            # NOTE: This method is optional if method `optimality` is implemented.
            ...

        def solve(self, batch, labels):
            # Conduct the inner-loop optimization
            ...
            return self  # optimized module

    # Get meta_params and data
    meta_params, data = ..., ...
    inner_net = InnerNet()

    # Solve for inner-loop process related to the meta-parameters
    optimal_inner_net = inner_net.solve(meta_params, *data)

    # Get outer-loss and solve for meta-gradient
    loss = outer_loss(optimal_inner_net)
    meta_grad = torch.autograd.grad(loss, meta_params)

If the optimization objective is to minimize/maximize an objective function, we offer an ``objective`` method interface to simplify the implementation.
Users only need to define the ``objective`` method, while TorchOpt will automatically analyze it for the stationary (optimality) condition from the KKT condition.

.. note::

    In ``__init__`` method, users need to define the inner parameters and meta-parameters.
    By default, :class:`nn.ImplicitMetaGradientModule` treats all tensors and modules from the method inputs as ``self.meta_parameters()`` / ``self.meta_modules()``.
    For example, statement ``self.yyy = xxx`` will assign ``xxx`` as a meta-parameter with name ``'yyy'`` if ``xxx`` is present in the method inputs (e.g., ``def __init__(self, xxx, ...): ...``).
    All tensors and modules defined in the ``__init__`` are regarded as ``self.parameters()`` / ``self.modules()``.
    Users can also register parameters and meta-parameters by calling ``self.register_parameter()`` and ``self.register_meta_parameter()`` respectively.

Linear System Solvers
---------------------

.. autosummary::

    torchopt.linear_solve.solve_cg
    torchopt.linear_solve.solve_inv
    torchopt.linear_solve.solve_normal_cg

Usually, the computation of implicit gradient involves the computation of the inverse Hessian matrix.
However, the high-dimensional Hessian matrix also makes direct computation intractable, and this is where linear solver comes into play.
By iteratively solving the linear system problem, we can calculate the inverse Hessian matrix up to some precision. We offer the `conjugate-gradient <https://arxiv.org/abs/1909.04630>`_ based solver and `neuman-series <https://arxiv.org/abs/1911.02590>`_ based solver.

Here is an example of the linear solver.

.. code-block:: python

    import torch
    from torchopt import linear_solve

    torch.manual_seed(42)
    A = torch.randn(3, 3)
    b = torch.randn(3)

    def matvec(x):
        return  torch.matmul(A, x)

    solve_fn = linear_solve.solve_normal_cg(atol=1e-5)
    solution = solve_fn(matvec, b)
    print(solution)

    solve_fn = linear_solve.solve_cg(atol=1e-5)
    solution = solve_fn(matvec, b)
    print(solution)

Users can also select the corresponding solver in functional and OOP APIs.

.. code-block:: python

    # For functional API
    @torchopt.diff.implicit.custom_root(
        functorch.grad(objective_fn, argnums=0),  # optimality function
        argnums=1,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    )
    def solve_fn(...):
        ...

    # For OOP API
    class InnerNet(
        torchopt.nn.ImplicitMetaGradientModule,
        linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    ):
        ...

Notebook Tutorial
-----------------

Check the notebook tutorial at `Implicit Differentiation <https://github.com/metaopt/torchopt/blob/main/tutorials/5_Implicit_Differentiation.ipynb>`_.
