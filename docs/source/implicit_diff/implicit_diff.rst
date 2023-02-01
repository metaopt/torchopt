.. _implicit_diff:

Implicit Gradient differentiation
=================================

Implicit differentiation
------------------------

.. image:: /_static/images/ig.png
    :scale: 60 %
    :align: center

Implicit differentiation is the task of differentiating a minimization problem's solution with respect to its inputs.
Namely, given

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) := \underset{\boldsymbol{\theta}}{\mathop{\operatorname{argmin}}} ~
    J^{\text{In}} (\boldsymbol{\phi},\boldsymbol{\theta}),

By treating the solution :math:`\boldsymbol{\theta}^{\prime}` as an implicit function of :math:`\boldsymbol{\phi}`, the idea of implicit differentiation is to directly get analytical best-response derivatives :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})` by implicit function theorem. This is suitable for algorithms when the inner-level optimal solution is achieved :math:`\frac{\partial J^{\text{In}} (\phi, \boldsymbol{\theta})}{\partial \theta} \rvert_{\theta = \theta^{\prime}} = 0` (so F in the figure means the solution is obtained by unrolled gradient steps) or reaches some stationary conditions :math:`F (\phi, \boldsymbol{\theta}^{\prime}) = 0`, such as `IMAML <https://arxiv.org/abs/1909.04630>`_ and `DEQ <https://arxiv.org/abs/1909.01377>`_.

Custom solvers
--------------

.. autosummary::

    torchopt.diff.implicit.custom_root

TorchOpt provides the ``custom_root`` decorators, for easily adding implicit differentiation on top of any existing solver (also called forward optimization). ``custom_root`` requires users to define the stationary conditions for the problem solution, e.g. KKT conditions, and will automatically calculate the gradient for backward gradient computation.

Here is an example of ``custom_root`` decorators, which is also the **functional API** for implicit gradient.

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
^^^^^^^

.. autosummary::

    torchopt.diff.implicit.nn.ImplicitMetaGradientModule

Coupled with PyTorch ``nn.Module``, we also design the OOP API ``ImplicitMetaGradientModule`` for implicit gradient. The core idea of ``ImplicitMetaGradientModule`` is to enable the gradient flow from `self.parameters()` (usually lower-level parameters) to `self.meta_parameters()` (usually the high-level parameters). Users need to define the forward process ``forward()``, a stationary function ``optimality()`` (or ``objective()``), and inner-loop optimization ``solve``.

Here is an example of the OOP API.

.. code-block:: python

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

    # Solve for inner-loop process related with the meta-parameters
    optimal_inner_net = inner_net.solve(meta_params, *data)

    # Get outer-loss and solve for meta-gradient
    loss = outer_loss(optimal_inner_net)
    meta_grad = torch.autograd.grad(loss, meta_params)

If the optimization objective is to minimize a loss function, we offer ``objective`` function to simplify the implementation. User only need to define the objective function, while TorchOpt will automatically analyze it for the stationary (optimality) condition.

.. note::

    In ``__init__`` function, users need to define the inner parameters and meta-parameters. By default, ``ImplicitMetaGradientModule`` treats all tensors and modules from input as ``self.meta_parameters()``, and all tensors and modules defined in the ``__init__`` are regarded as ``self.parameters()``. Users can also register `self.parameters()` and `self.meta_parameters()` by calling ``self.register_parameter()`` and ``self.register_meta_parameter()`` respectively.

Linear System Solvers
---------------------

.. autosummary::

    torchopt.linear_solve.solve_cg
    torchopt.linear_solve.solve_inv
    torchopt.linear_solve.solve_normal_cg

Usually, the computation of implicit gradient involves the computation of inverse Hessian matrix. However, the high-dimensional Hessian matrix also makes direct computation intractable, and this is where linear solver comes into play. By iteratively solving the linear system problem, we can calculate inverse Hessian matrix up to some precision. We offer the `conjugate-gradient <https://arxiv.org/abs/1909.04630>`_ based solver and `neuman-series <https://arxiv.org/abs/1911.02590>`_ based solver.

Here is an example of the linear solver.

.. code-block:: python

    from torchopt import linear_solve

    torch.random.seed(42)
    A = torch.random.randn(3, 3)
    b = torch.random.randn(3)

    def matvec_A(x):
        return  torch.dot(A, x)

    sol = linear_solve.solve_normal_cg(matvec_A, b, tol=1e-5)
    print(sol)

    sol = linear_solve.solve_cg(matvec_A, b, tol=1e-5)
    print(sol)

User can also select corresponding solver in functional and OOP API.

.. code-block:: python

    # For functional API
    @torchopt.diff.implicit.custom_root(
        functorch.grad(imaml_objective, argnums=0),  # optimality function
        argnums=1,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    )

    # For OOP API
    class InnerNet(
        torchopt.nn.ImplicitMetaGradientModule,
        linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    )

Notebook Tutorial
-----------------
Check notebook tutorial at `Implicit Differentiation <https://github.com/metaopt/torchopt/blob/main/tutorials/5_Implicit_Differentiation.ipynb>`_.
