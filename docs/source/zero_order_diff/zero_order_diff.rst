Zero-order Gradient Differentiation
===================================

Evolutionary Strategy
---------------------

.. image:: /_static/images/zero_order.png
    :scale: 60 %
    :align: center

When the inner-loop process is non-differentiable or one wants to eliminate the heavy computation burdens in the previous two modes (brought by Hessian), one can choose Zeroth-order differentiation. Zeroth-order differentiation typically gets gradients based on zero-order estimation, such as finite-difference, or `Evolutionary Strategy <https://arxiv.org/abs/1703.03864>`_ (ES).  `ES-MAML <https://arxiv.org/pdf/1910.01215.pdf>`_, and `NAC <https://arxiv.org/abs/2106.02745>`_, successfully solve the non-differentiable optimization problem based on ES.

TorchOpt offers API for ES-based differentiation. Instead of optimizing the objective :math:`F`, ES optimizes a Gaussian smoothing objective defined as :math:`\tilde{f}_{\sigma} (\theta) = \mathbb{E}_{{z} \sim \mathcal{N}( {0}, {I}_d )} [ f ({\theta} + \sigma \, z) ]`, where :math:`\sigma` denotes precision. The gradient of such objective is :math:`\nabla_\theta \tilde{f}_{\sigma} (\theta) = \frac{1}{\sigma} \mathbb{E}_{{z} \sim \mathcal{N}( {0}, {I}_d )} [ f({\theta} + \sigma \, z) \cdot z ]`. Based on such technique, one can treat the bi-level process as a whole to calculate the meta-gradient based on pure forward process. Refer to `ES-MAML <https://arxiv.org/pdf/1910.01215.pdf>`_ for more explanations.


Decorators
----------

.. autosummary::

    torchopt.diff.zero_order.zero_order

Similar with the implicit gradient, we also use the decorator for ES methods.

Functional API
^^^^^^^^^^^^^^

The basic functional API is ``torchopt.diff.zero_order.zero_order``, which is used as the decorator for the forward process zero-order gradient procedures. Users are required to implement the noise sampling function, which will be used as the input of zero_order decorator. Here we show the specific meaning for each parameter used in the decorator.

- ``distribution`` for noise sampling distribution. The distribution :math:`\lambda` should be spherical symmetric and with a constant variance of :math:`1` for each element. I.e.:
    - Spherical symmetric: :math:`\mathbb{E}_{\boldsymbol{z} \sim \lambda} [ \boldsymbol{z} ] = \boldsymbol{0}`.
    - Constant variance of :math:`1` for each element: :math:`\mathbb{E}_{\boldsymbol{z} \sim \lambda} [ {\lvert \boldsymbol{z}_i \rvert}^2 ] = 1`.
    - An easy example is normal distribution
- ``method`` for different kind of algorithms, we support `'naive'` (`ES RL <https://arxiv.org/abs/1703.03864>`_), `'forward'` (`Forward-FD <http://proceedings.mlr.press/v80/choromanski18a/choromanski18a.pdf>`_), and `'antithetic'` (`antithetic <https://d1wqtxts1xzle7.cloudfront.net/75609515/coredp2011_1web-with-cover-page-v2.pdf?Expires=1670215467&Signature=RfP~mQhhhI7aGknwXbRBgSggFrKuNTPYdyUSdMmfTxOa62QoOJAm-Xhr3F1PLyjUQc2JVxmKIKGGuyYvyfCTpB31dfmMtuVQxZMWVF-SfErTN05SliC93yjA1x1g2kjhn8bkBFdQqGl~1RQSKnhj88BakgSeDNzyCxwbD5VgR89BXRs4YIK5RBIKYtgLhoyz5jar7wHS3TJhRzs3WNeTIAjAmLqJ068oGFZ0Jr7maGquTe3w~8LEEIprJ6cyCMc6b1UUJkmwjNq0RLTVbxgFjfi4Z9kyxyJB9IOS1J25OOON4jfwh5JlXS7MVskuONUyHJim1TQ8OwCraKlBsQLPQw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA>`_).
- ``argnums`` specifies which parameter we want to trace the meta-gradient.
- ``num_samples`` specifies how many times we want to conduct the sampling.
- ``sigma`` is for precision. This is the scaling factor for the sampling distribution.

We show the pseudo code in the following part.

.. code-block:: python

    # Functional API for zero-order differentiation
    # 1. Customize the noise distribution via a distribution class
    class Distribution:
        def sample(self, sample_shape=torch.Size()):
            # Sampling function for noise
            # NOTE: The distribution should be spherical symmetric and with a constant variance of 1.
            ...
            return noise_batch

    distribution = Distribution()

    # 2. Customize the noise distribution via a sampling function
    def distribution(sample_shape=torch.Size()):
        # Sampling function for noise
        # NOTE: The distribution should be spherical symmetric and with a constant variance of 1.
        ...
        return noise_batch

    # 3. Distribution can also be an instance of `torch.distributions.Distribution`, e.g., `torch.distributions.Normal(...)`
    distribution = torch.distributions.Normal(loc=0, scale=1)

    # Decorator that wraps the function
    @torchopt.diff.zero_order(distribution=distribution, method='naive', argnums=0, num_samples=100, sigma=0.01)
    def forward(params, data):
        # Forward optimization process for params
        ...
        return objective  # the returned tensor should be a scalar tensor

    # Define params and get data
    params, data = ..., ...

    # Forward pass
    loss = forward(params, data)
    # Backward pass using zero-order differentiation
    grads = torch.autograd.grad(loss, params)

OOP API
^^^^^^^

.. autosummary::

    torchopt.diff.zero_order.nn.ZeroOrderGradientModule

Coupled with PyTorch ``nn.Module``, we also design the OOP API ``ZeroOrderGradientModule`` for ES. The core idea of ``ZeroOrderGradientModule`` is to enable the gradient flow Forward process  to `self.parameters()` (can be the meta-parameters when calculate meta-gradient). Users need to define the forward process zero-order gradient procedures ``forward()`` and a noise sampling function ``sample()``.

.. code-block:: python

    from torchopt.nn import ZeroOrderGradientModule

    # Inherited from the class ZeroOrderGradientModule
    # Optionally specify the `method` and/or `num_samples` and/or `sigma` used for sampling
    class Net(ZeroOrderGradientModule, method='naive', num_samples=100, sigma=0.01):
        def __init__(self, ...):
            ...

        def forward(self, batch):
            # Forward process
            ...
            return objective  # the returned tensor should be a scalar tensor

        def sample(self, sample_shape=torch.Size()):
            # Generate a batch of noise samples
            # NOTE: The distribution should be spherical symmetric and with a constant variance of 1.
            ...
            return noise_batch

    # Get model and data
    net = Net(...)
    data = ...

    # Forward pass
    loss = Net(data)
    # Backward pass using zero-order differentiation
    grads = torch.autograd.grad(loss, net.parameters())

Notebook Tutorial
-----------------
For more details, check notebook tutorial at `zero order <https://github.com/metaopt/torchopt/blob/main/tutorials/6_Zero_Order_Differentiation.ipynb>`_.
