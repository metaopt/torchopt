Explicit Gradient differentiation
=================================

Explicit gradient
-----------------

The idea of Explicit Gradient is to treat the gradient step as a differentiable function and try to backpropagate through the unrolled optimization path.
Namely, given

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) := \boldsymbol{\theta}^{0} - \alpha \sum_{i=0}^{K-1} \nabla_{\boldsymbol{\theta}^{i}} J^{\text{In}} (\boldsymbol{\phi},\boldsymbol{\theta}^{i}),

we would like to compute the Gradient :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})`.
This is usually done by autodiff through an inner optimization's unrolled iterates.

Differentiable meta-optimizers
------------------------------

.. autosummary::

    torchopt.MetaOptimizer
    torchopt.MetaAdam
    torchopt.MetaSGD
    torchopt.MetaRMSProp
    torchopt.MetaAdamW



.. code-block:: python

    # Low-level API
    optim = torchopt.MetaOptimizer(net, torchopt.sgd(lr=1.0))

    # High level API
    optim = torchopt.MetaSGD(net, lr=1.0)

General Utilities
-----------------

.. autosummary::

    torchopt.utils.extract_state_dict
    torchopt.utils.recover_state_dict
    torchopt.utils.stop_gradient


We provide the ``torchopt.extract_state_dict`` and ``torchopt.recover_state_dict`` functions to extract and restore the state of network and optimizer. By default, the extracted state dictionary is a reference (this design is for accumulating gradient of multi-task batch training, MAML for example). You can also set ``by='copy'`` to extract the copy of state dictionary or set ``by='deepcopy'`` to have a detached copy.

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


OOP API
-------


.. code-block:: python

    # Define meta and inner parameters
    meta_params = ...
    model = ...
    # Define differentiable optimizer
    opt = torchopt.MetaAdam(model)

    for iter in range(iter_times):
        # Perform the inner update
        loss = inner_loss(model, meta_params)
        opt.step(loss)

    loss = outer_loss(model, meta_params)
    loss.backward()



Functional API
--------------

.. code-block:: python

    opt = torchopt.adam()
    # Define meta and inner parameters
    meta_params = ...
    fmodel, params = make_functional(model)
    # Initialize optimizer state
    state = opt.init(params)

    for iter in range(iter_times):
        loss = inner_loss(fmodel, params, meta_params)
        grads = torch.autograd.grad(loss, params)
        # Apply non-inplace parameter update
        updates, state = opt.update(grads, state, inplace=False)
        params = torchopt.apply_updates(params, updates)

    loss = outer_loss(fmodel, params, meta_params)
    meta_grads = torch.autograd.grad(loss, meta_params)
