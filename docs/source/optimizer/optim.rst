Optimizers
==========

The design of TorchOpt follows the philosophy of functional programming. Aligned with ``functorch``, users can conduct functional style programming with models, optimizers and training in PyTorch.

Classic Optimizers
------------------

.. autosummary::

    torchopt.Optimizer
    torchopt.Adam
    torchopt.SGD
    torchopt.RMSProp
    torchopt.AdamW


PyTorch-Like API
~~~~~~~~~~~~~~~~

We designed base class ``torchopt.Optimizer`` that has the same interface as ``torch.optim.Optimizer``. We offer origin PyTorch APIs
(e.g. ``zero_grad()`` or ``step()``) for traditional PyTorch users.

.. code-block:: python

    net = Net()  # init
    loader = Loader()
    optimizer = torchopt.Adam(net.parameters())

    xs, ys = next(loader)             # get data
    pred = net(xs)                    # forward
    loss = F.cross_entropy(pred, ys)  # compute loss

    optimizer.zero_grad()             # zero gradients
    loss.backward()                   # backward
    optimizer.step()                  # step updates

Accelerated Optimizer
~~~~~~~~~~~~~~~~~~~~~

Users can use accelerated optimizer by setting the ``use_accelerated_op`` as ``True``. Currently we only support the Adam optimizer.

.. code-block:: python

    # Check whether the `accelerated_op` is available:
    torchopt.accelerated_op_available(torch.device('cpu'))

    net = Net(1).cuda()
    optim = torchopt.Adam(net.parameters(), lr=1.0, use_accelerated_op=True)

Functional Optimizers
---------------------

.. autosummary::

    torchopt.FuncOptimizer
    torchopt.adam
    torchopt.sgd
    torchopt.rmsprop
    torchopt.adamw

Apply Parameter Updates
-----------------------

.. autosummary::

    torchopt.apply_updates

Optax-Like API
~~~~~~~~~~~~~~

For those users who prefer fully functional programming, we offer Optax-Like API by passing gradients and optimizers states to the optimizer function. Here is an example coupled with ``functorch``:

.. code-block:: python

    class Net(nn.Module): ...

    class Loader(DataLoader): ...

    net = Net()  # init
    loader = Loader()
    optimizer = torchopt.adam()

    model, params = functorch.make_functional(net)           # use functorch extract network parameters
    opt_state = optimizer.init(params)                       # init optimizer

    xs, ys = next(loader)                                    # get data
    pred = model(params, xs)                                 # forward
    loss = F.cross_entropy(pred, ys)                         # compute loss

    grads = torch.autograd.grad(loss, params)                # compute gradients
    updates, opt_state = optimizer.update(grads, opt_state)  # get updates
    params = torchopt.apply_updates(params, updates)         # update network parameters


Combining Optimizers
--------------------

.. autosummary::

    torchopt.chain

In the designing of TorchOpt, we treat these functions as derivations of ``combine.chain``. So we can build our own chain like ``combine.chain(clip.clip_grad_norm(max_norm=1.), sgd(lr=1., requires_grad=True))`` to clip the gradient and update parameters using ``sgd``.

Optimizer Hooks
---------------

.. autosummary::

    torchopt.register_hook
    torchopt.hook.zero_nan_hook
    torchopt.hook.nan_to_num_hook

Register hook to the first-order gradients. During the backpropagation, the NaN gradients will be set to 0, which will have a similar effect to the first solution but much slower.

.. code-block:: python

    impl = torchopt.chain(torchopt.hook.register_hook(torchopt.hook.zero_nan_hook), torchopt.adam(1e-1))
    inner_optim = torchopt.MetaOptimizer(net, impl)

Optimizer Schedules
-------------------

.. autosummary::

    torchopt.schedule.linear_schedule
    torchopt.schedule.polynomial_schedule

TorchOpt also provides implementation of learning rate scheduler, which can be used as:

.. code-block:: python

    functional_adam = torchopt.adam(
        lr=torchopt.schedule.linear_schedule(
            init_value=1e-3, end_value=1e-4, transition_steps=10000, transition_begin=2000
        )
    )

    adam = torchopt.Adam(
        net.parameters(),
        lr=torchopt.schedule.linear_schedule(
            init_value=1e-3, end_value=1e-4, transition_steps=10000, transition_begin=2000
        ),
    )
