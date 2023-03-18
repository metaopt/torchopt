Implicit Gradient Differentiation
=================================

.. currentmodule:: torchopt.diff.implicit

Implicit Differentiation
------------------------

.. image:: /_static/images/implicit-gradient.png
    :width: 80%
    :align: center

Implicit differentiation is the task of differentiating through the solution of an optimization problem satisfying a mapping function :math:`T` capturing the optimality conditions of the problem.
The simplest example is to differentiate through the solution of a minimization problem with respect to its inputs.
Namely, given

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) \triangleq \underset{\boldsymbol{\theta}}{\mathop{\operatorname{argmin}}} ~ \mathcal{L}^{\text{in}} (\boldsymbol{\phi},\boldsymbol{\theta}).

By treating the solution :math:`\boldsymbol{\theta}^{\prime}` as an implicit function of :math:`\boldsymbol{\phi}`, the idea of implicit differentiation is to directly get analytical best-response derivatives :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})` by the implicit function theorem.

Root Finding
~~~~~~~~~~~~

This is suitable for algorithms when the inner-level optimality conditions :math:`T` is defined by a root of a function, such as:

.. math::

    T (\boldsymbol{\phi}, \boldsymbol{\theta}) = \frac{ \partial \mathcal{L}^{\text{in}} (\boldsymbol{\phi}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}, \qquad T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})) = \left. \frac{ \partial \mathcal{L}^{\text{in}} (\boldsymbol{\phi}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \right\rvert_{\boldsymbol{\theta} = \boldsymbol{\theta}^{\prime}} = \boldsymbol{0}.

In `IMAML <https://arxiv.org/abs/1909.04630>`_, the function :math:`F` in the figure means the inner-level optimal solution is obtained by unrolled gradient update:

.. math::

    \boldsymbol{\theta}_{k + 1} = F (\boldsymbol{\phi}, \boldsymbol{\theta}_k) = \boldsymbol{\theta}_k - \alpha \nabla_{\boldsymbol{\theta}_k} \mathcal{L}^{\text{in}} (\boldsymbol{\phi}, \boldsymbol{\theta}_k).

Fixed-point Iteration
~~~~~~~~~~~~~~~~~~~~~

Sometimes the inner-level optimal solution can also be achieved by fixed point where the optimality :math:`T` takes the form:

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) = F (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})) \quad \Longleftrightarrow \quad T (\boldsymbol{\phi}, \boldsymbol{\theta}) = F (\boldsymbol{\phi}, \boldsymbol{\theta}) - \boldsymbol{\theta}, \quad T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})) = \boldsymbol{0}.

In `DEQ <https://arxiv.org/abs/1909.01377>`_, the function :math:`F` in the figure means the inner-level optimal solution is obtained by fixed point update:

.. math::

    \boldsymbol{\theta}_{k + 1} = F (\boldsymbol{\phi}, \boldsymbol{\theta}_k).

This can be seen as a particular case of root of function by defining the optimality function as :math:`T (\boldsymbol{\phi}, \boldsymbol{\theta}) = F (\boldsymbol{\phi}, \boldsymbol{\theta}) - \boldsymbol{\theta}`.
This can be implemented with:

.. code-block:: python

    def fixed_point_function(phi: TensorTree, theta: TensorTree) -> TensorTree:
        ...
        return new_theta

    # A root function can be derived from the fixed point function
    def root_function(phi: TensorTree, theta: TensorTree) -> TensorTree:
        new_theta = fixed_point_function(phi, theta)
        return torchopt.pytree.tree_sub(new_theta, theta)

Custom Solvers
--------------

.. autosummary::

    torchopt.diff.implicit.custom_root

Let :math:`T (\boldsymbol{\phi}, \boldsymbol{\theta}): \mathbb{R}^n \times \mathbb{R}^d \to \mathbb{R}^d` be a user-provided mapping function, that captures the optimality conditions of a problem.
An optimal solution, denoted :math:`\boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})`, should be a root of :math:`T`:

.. math::

    T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime}(\boldsymbol{\phi})) = \boldsymbol{0}.

We can see :math:`\boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})` as an implicitly defined function of :math:`\boldsymbol{\phi} \in \mathbb{R}^n`, i.e., :math:`\boldsymbol{\theta}^{\prime}: \mathbb{R}^n \rightarrow \mathbb{R}^d`.
More precisely, from the `implicit function theorem <https://en.wikipedia.org/wiki/Implicit_function_theorem>`_, we know that for :math:`(\boldsymbol{\phi}_0, \boldsymbol{\theta}^{\prime}_0)` satisfying :math:`T (\boldsymbol{\phi}_0, \boldsymbol{\theta}^{\prime}_0) = \boldsymbol{0}` with a continuously differentiable :math:`T`, if the Jacobian :math:`\nabla_{\boldsymbol{\theta}^{\prime}} T` evaluated at :math:`(\boldsymbol{\phi}_0, \boldsymbol{\theta}^{\prime}_0)` is a square invertible matrix, then there exists a function :math:`\boldsymbol{\theta}^{\prime} (\cdot)` defined on a neighborhood of :math:`\boldsymbol{\phi}_0` such that :math:`\boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}_0) = \boldsymbol{\theta}^{\prime}_0`.
Furthermore, for all :math:`\boldsymbol{\phi}` in this neighborhood, we have that :math:`T (\boldsymbol{\phi}_0, \boldsymbol{\theta}^{\prime}_0) = \boldsymbol{0}` and :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})` exists. Using the chain rule, the Jacobian :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime}(\boldsymbol{\phi})` satisfies:

.. math::

    \frac{d T}{d \boldsymbol{\phi}} = \underbrace{\nabla_{\boldsymbol{\theta}^{\prime}} T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime}(\boldsymbol{\phi}))}_{\frac{\partial T}{\partial \boldsymbol{\theta}^{\prime}}} \underbrace{\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})}_{\frac{d \boldsymbol{\theta}^{\prime}}{d \boldsymbol{\phi}}} + \underbrace{\nabla_{\boldsymbol{\phi}} T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}))}_{\frac{\partial T}{\partial \boldsymbol{\phi}}} = \boldsymbol{0}. \qquad ( T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime}) = \boldsymbol{0} = \text{const})

Computing :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime}(\boldsymbol{\phi})` therefore boils down to the resolution of the linear system of equations

.. math::

    \underbrace{\nabla_{\boldsymbol{\theta}^{\prime}} T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime}(\boldsymbol{\phi}))}_{A \in \mathbb{R}^{d \times d}} \underbrace{\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})}_{J \in \mathbb{R}^{d \times n}} = \underbrace{- \nabla_{\boldsymbol{\phi}} T (\boldsymbol{\phi}, \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}))}_{B \in \mathbb{R}^{d \times n}}.

TorchOpt provides a decorator function :func:`custom_root`, for easily adding implicit differentiation on top of any existing inner optimization solver (also called forward optimization).
The :func:`custom_root` decorator requires users to define the stationary conditions for the problem solution (e.g., `KKT conditions <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_) and will automatically calculate the gradient for backward gradient computation.

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
