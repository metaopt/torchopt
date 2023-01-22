Zero-order Gradient differentiation
===================================

Zeroth-order differentiation
----------------------------

Zeroth-order differentiation typically gets gradients based on zero-order estimation, when the inner-loop process is non-differentiable or one wants to eliminate the heavy computation burdens in the previous two modes (brought by Hessian), one can choose zeroth-order differentiation.

.. math::

    \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi}) := \mathbb{E}_{\boldsymbol{z} \sim \mathcal{N} ( \boldsymbol{0}, \boldsymbol{I}_d )} [ \boldsymbol{\theta} + \sigma \boldsymbol{z} ]


we would like to compute the Gradient :math:`\nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}^{\prime} (\boldsymbol{\phi})`.
This is usually done either by finite-difference, or Evolutionary Strategy (ES).


Decorators
----------

.. autosummary::

    torchopt.diff.zero_order.zero_order


OOP API
-------


.. code-block:: python

    # Customize the noise sampling function in ES
    def sample(sample_shape):
        ...
        return sample_noise

    # Specify the method and parameter of ES
    @torchopt.diff.zero_order(method, sample)
    def forward(params, batch, labels):
        # Forward process
        return output



Functional API
--------------

.. code-block:: python

    class ESModule(torchopt.nn.ZeroOrderGradientModule):
        def sample(self, sample_shape):
            # Customize the noise sampling function in ES
            ...
            return sample_noise

        def forward(self, batch, labels):
            # Forward process
            ...
            return output
