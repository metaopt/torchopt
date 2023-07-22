Optimizers
==========

.. currentmodule:: torchopt

The core design of TorchOpt follows the philosophy of functional programming.
Aligned with |functorch|_, users can conduct functional-style programming with models, optimizers, and training in PyTorch.
We first introduce our functional optimizers, which treat the optimization process as a functional transformation.

.. |functorch| replace:: ``functorch``
.. _functorch: https://pytorch.org/functorch

Functional Optimizers
---------------------

Currently, TorchOpt supports 4 functional optimizers: :func:`sgd`, :func:`adam`, :func:`rmsprop`, and :func:`adamw`.

.. autosummary::

    torchopt.FuncOptimizer
    torchopt.adadelta
    torchopt.adagrad
    torchopt.adam
    torchopt.adamw
    torchopt.adamax
    torchopt.radam
    torchopt.rmsprop
    torchopt.sgd

Apply Parameter Updates
-----------------------

TorchOpt offers functional API by passing gradients and optimizer states to the optimizer function to apply updates.

.. autosummary::

    torchopt.apply_updates

Here is an example of functional optimization coupled with |functorch|_:

.. code-block:: python

    class Net(nn.Module): ...

    class Loader(DataLoader): ...

    net = Net()  # init
    loader = Loader()
    optimizer = torchopt.adam(lr)

    model, params = functorch.make_functional(net)           # use functorch extract network parameters
    opt_state = optimizer.init(params)                       # init optimizer

    xs, ys = next(loader)                                    # get data
    pred = model(params, xs)                                 # forward
    loss = F.cross_entropy(pred, ys)                         # compute loss

    grads = torch.autograd.grad(loss, params)                # compute gradients
    updates, opt_state = optimizer.update(grads, opt_state)  # get updates
    params = torchopt.apply_updates(params, updates)         # update network parameters

We also provide a wrapper :class:`torchopt.FuncOptimizer` to make maintaining the optimizer state easier:

.. code-block:: python

    net = Net()  # init
    loader = Loader()
    optimizer = torchopt.FuncOptimizer(torchopt.adam())      # wrap with `torchopt.FuncOptimizer`

    model, params = functorch.make_functional(net)           # use functorch extract network parameters

    for xs, ys in loader:                                    # get data
        pred = model(params, xs)                             # forward
        loss = F.cross_entropy(pred, ys)                     # compute loss

        params = optimizer.step(loss, params)                # update network parameters

Classic OOP Optimizers
----------------------

Combined with the functional optimizers above, we can define our classic OOP optimizers.
We designed a base class :class:`torchopt.Optimizer` that has the same interface as |torch.optim.Optimizer|_.
We offer original PyTorch APIs (e.g., ``zero_grad()`` or ``step()``) for traditional PyTorch-like (OOP) parameter update.

.. |torch.optim.Optimizer| replace:: ``torch.optim.Optimizer``
.. _torch.optim.Optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer

.. autosummary::

    torchopt.Optimizer
    torchopt.AdaDelta
    torchopt.Adadelta
    torchopt.AdaGrad
    torchopt.Adagrad
    torchopt.Adam
    torchopt.AdamW
    torchopt.AdaMax
    torchopt.Adamax
    torchopt.RAdam
    torchopt.RMSProp
    torchopt.SGD


By combining low-level API :class:`torchopt.Optimizer` with the previous functional optimizer, we can achieve high-level API:

.. code-block:: python

    learning_rate = 1.0
    # High-level API
    optim = torchopt.Adam(net.parameters(), lr=learning_rate)
    # which can be achieved by low-level API:
    optim = torchopt.Optimizer(net.parameters(), torchopt.adam(lr=learning_rate))

Here is an example of PyTorch-like APIs:

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

Combining Transformation
------------------------

Users always need to conduct multiple gradient transformations (functions) before the final update.
In the designing of TorchOpt, we treat these functions as derivations of :func:`torchopt.chain`.
So we can build our own chain like ``torchopt.chain(torchopt.clip_grad_norm(max_norm=1.), torchopt.sgd(lr=1., moment_requires_grad=True))`` to clip the gradient and update parameters using :func:`sgd`.

.. autosummary::

    torchopt.chain

.. note::

    :func:`torchopt.chain` will sequentially conduct transformations, so the order matters.
    For example, we need to first conduct gradient normalization and then conduct the optimizer step.
    The order should be (clip, sgd) in :func:`torchopt.chain` function.


Here is an example of chaining :func:`torchopt.clip_grad_norm` and :func:`torchopt.adam` for functional optimizer and OOP optimizer.

.. code-block:: python

    func_optimizer = torchopt.chain(torchopt.clip_grad_norm(max_norm=2.0), torchopt.adam(1e-1))
    oop_optimizer = torchopt.Optimizer(net.parameters() func_optimizer)

Optimizer Hooks
---------------

Users can also add optimizer hook to control the gradient flow.

.. autosummary::

    torchopt.hook.register_hook
    torchopt.hook.zero_nan_hook
    torchopt.hook.nan_to_num_hook

For example, :func:`torchopt.hook.zero_nan_hook` registers hook to the first-order gradients.
During the backpropagation, the **NaN** gradients will be set to 0.
Here is an example of such operation coupled with :func:`torchopt.chain`.

.. code-block:: python

    impl = torchopt.chain(torchopt.hook.register_hook(torchopt.hook.zero_nan_hook), torchopt.adam(1e-1))

Optimizer Schedules
-------------------

TorchOpt also provides implementations of learning rate schedulers, which can be used to control the learning rate during the training process.
TorchOpt mainly offers the linear learning rate scheduler and the polynomial learning rate scheduler.

.. autosummary::

    torchopt.schedule.linear_schedule
    torchopt.schedule.polynomial_schedule

Here is an example of combining optimizer with learning rate scheduler.

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

Notebook Tutorial
-----------------

Check the notebook tutorial at `Functional Optimizer <https://github.com/metaopt/torchopt/blob/main/tutorials/1_Functional_Optimizer.ipynb>`_.
