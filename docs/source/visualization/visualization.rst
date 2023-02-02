Visualization
=============

.. currentmodule:: torchopt.visual

In `PyTorch <https://pytorch.org>`_, if the attribute ``requires_grad`` of a tensor is :data:`True`, the computation graph will be created if we use the tensor to do any operations.
The computation graph is implemented like a link list -- ``Tensors`` are nodes and they are linked by their attribute ``gran_fn``.
`PyTorchViz <https://github.com/szagoruyko/pytorchviz>`_ is a Python package that uses `Graphviz <https://graphviz.org>`_ as a backend for plotting computation graphs.
TorchOpt uses PyTorchViz as the blueprint and provides more easy-to-use visualization functions on the premise of supporting all its functions.

------

Usage
-----

Let's start with a simple multiplication computation graph.
We declared the variable ``x`` with the flag ``requires_grad=True`` and compute ``y = 2 * x``. Then we visualize the computation graph of ``y``.

We provide the function :func:`make_dot` which takes a tensor as input.
The visualization code is shown as follows:

.. code-block:: python

    from IPython.display import display
    import torch
    import torchopt


    x = torch.tensor(1.0, requires_grad=True)
    y = 2 * x
    display(torchopt.visual.make_dot(y))

.. image:: /_static/images/visualization-fig1.svg
    :width: 20%
    :align: center

The figure shows ``y`` is connected by the multiplication edge.
The gradient of ``y`` will flow through the multiplication backward function and then accumulate on ``x``.
Note that we pass a dictionary for adding node labels.

To add auxiliary notes to the computation graph, we can pass a dictionary as argument ``params`` to :func:`make_dot`.
The keys are the notes which would be shown in the computation figure and the values are the tensors that need to be noted.
So the code above can be modified as follows:

.. code-block:: python

    from IPython.display import display
    import torch
    import torchopt


    x = torch.tensor(1.0, requires_grad=True)
    y = 2 * x
    display(torchopt.visual.make_dot(y, params={'x': x, 'y': y}))

Then let's plot a neural network.
Note that we can pass the generator returned by the method ``named_parameters`` for adding node labels.

.. code-block:: python

    from IPython.display import display
    import torch
    from torch import nn
    import torchopt


    class Net(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, 1, bias=True)

        def forward(self, x):
            return self.fc(x)


    dim = 5
    batch_size = 2
    net = Net(dim)
    xs = torch.ones((batch_size, dim))
    ys = torch.ones((batch_size, 1))
    pred = net(xs)
    loss = F.mse_loss(pred, ys)

    display(torchopt.visual.make_dot(loss, params=(net.named_parameters(), {'loss': loss})))

.. image:: /_static/images/visualization-fig2.svg
    :width: 45%
    :align: center

The computation graph of meta-learning algorithms will be much more complex.
Our visualization tool allows users to take as input the extracted network state for better visualization.

.. code-block:: python

    from IPython.display import display
    import torch
    from torch import nn
    import torchopt

    class MetaNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, 1, bias=True)

        def forward(self, x, meta_param):
            return self.fc(x) + meta_param


    dim = 5
    batch_size = 2
    net = MetaNet(dim)

    xs = torch.ones((batch_size, dim))
    ys = torch.ones((batch_size, 1))

    optimizer = torchopt.MetaSGD(net, lr=1e-3)
    meta_param = torch.tensor(1.0, requires_grad=True)

    # Set enable_visual
    net_state_0 = torchopt.extract_state_dict(net, enable_visual=True, visual_prefix='step0.')

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, ys)
    optimizer.step(loss)

    # Set enable_visual
    net_state_1 = torchopt.extract_state_dict(net, enable_visual=True, visual_prefix='step1.')

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))

    # Draw computation graph
    display(
        torchopt.visual.make_dot(
            loss, [net_state_0, net_state_1, {'meta_param': meta_param, 'loss': loss}]
        )
    )

.. image:: /_static/images/visualization-fig3.svg
    :width: 65%
    :align: center

Notebook Tutorial
-----------------

Check the notebook tutorial at `Visualization <https://github.com/metaopt/torchopt/blob/main/tutorials/2_Visualization.ipynb>`_.
