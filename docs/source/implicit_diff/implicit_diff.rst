.. _implicit_diff:

Implicit Gradient differentiation
=================================

Argmin differentiation
----------------------

Argmin differentiation is the task of differentiating a minimization problem's solution with respect to its inputs.
Namely, given

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) := \underset{\boldsymbol{\theta}^{\prime}}{\mathop{\operatorname{argmin}}} ~
    J^{\text{In}} (\boldsymbol{\phi},\boldsymbol{\theta}^{i}),

we would like to compute the Gradient :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})`.
This is usually done either by implicit differentiation or by autodiff through an algorithm's unrolled iterates.

Custom solvers
--------------

.. autosummary::

    torchopt.diff.implicit.custom_root

TorchOpt provides the ``custom_root`` decorators, for easily adding implicit differentiation on top of any existing solver.

.. .. topic:: Examples

..     .. literalinclude:: implicit_diff.py
..         :language: python
..         :linenos:

.. code-block:: python

    net = Net()
    x = nn.Parameter(torch.tensor(2.0), requires_grad=True)

    optim = torchopt.MetaAdam(net, lr=1.0)

    # Get the reference of state dictionary
    init_net_state = torchopt.extract_state_dict(net, by='reference')
    init_optim_state = torchopt.extract_state_dict(optim, by='reference')
    # If set `detach_buffers=True`, the parameters are referenced as references while buffers are detached copies
    init_net_state = torchopt.extract_state_dict(net, by='reference', detach_buffers=True)

    # Set `copy` to get the copy of state dictionary
    init_net_state_copy = torchopt.extract_state_dict(net, by='copy')
    init_optim_state_copy = torchopt.extract_state_dict(optim, by='copy')

    # Set `deepcopy` to get the detached copy of state dictionary
    init_net_state_deepcopy = torchopt.extract_state_dict(net, by='deepcopy')
    init_optim_state_deepcopy = torchopt.extract_state_dict(optim, by='deepcopy')

    # Conduct 2 inner-loop optimization
    for i in range(2):
        inner_loss = net(x)
        optim.step(inner_loss)

    print(f'a = {net.a!r}')

    # Recover and reconduct 2 inner-loop optimization
    torchopt.recover_state_dict(net, init_net_state)
    torchopt.recover_state_dict(optim, init_optim_state)

    for i in range(2):
        inner_loss = net(x)
        optim.step(inner_loss)

    print(f'a = {net.a!r}')  # the same result


Linear System Solvers
---------------------

.. autosummary::

    torchopt.linear_solve.cg.solve_cg
    torchopt.linear_solve.inv.solve_inv
    torchopt.linear_solve.normal_cg.solve_normal_cg


Indirect solvers iteratively solve the linear system up to some precision. Example:

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


OOP API
-------

.. code-block:: python

    class Module(torchopt.nn.ImplicitMetaGradientModule):
        def __init__(self, meta_module, ...):
            ...
        def forward(self, x):
            # Forward process
            ...
        def optimality(self, batch, labels):
            # Stationary condition construction
            ...
        def solve(self, batch, labels):
            # Forward optimization process
            ...
            return self



Functional API
--------------

.. code-block:: python

    def stationary(params, meta_params, batch, labels):
        # Stationary condition construction
        ...
        return stationary condition

    @torchopt.diff.implicit.custom_root(stationary)
    def solve(params, meta_params, batch, labels):
        # Forward optimization process
        ...
        return optimal_params
