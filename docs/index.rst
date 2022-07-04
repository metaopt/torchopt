:github_url: https://github.com/metaopt/TorchOpt/tree/main/docs

TorchOpt
--------

**TorchOpt** is a high-performance optimizer library built upon `PyTorch <https://pytorch.org/>`_ for easy implementation of functional optimization and gradient-based meta-learning. It consists of two main features:

*   TorchOpt provides functional optimizer which enables `JAX-like <https://github.com/google/jax>`_ composable functional optimizer for PyTorch. With TorchOpt, one can easily conduct neural network optimization in PyTorch with functional style optimizer, similar to  `Optax <https://github.com/deepmind/optax>`_ in JAX.
*   With the desgin of functional programing, TorchOpt provides efficient, flexible, and easy-to-implement differentiable optimizer for gradient-based meta-learning research. It largely reduces the efforts required to implement sophisticated meta-learning algorithms.

Installation
------------

Requirements

(Optional) For visualizing computation graphs
`Graphviz <https://graphviz.org/download/>`_ (for Linux users use ``apt/yum install graphviz`` or ``conda install -c anaconda python-graphviz``)

.. code-block:: bash

    pip install TorchOpt

You can also build shared libraries from source, use:

.. code-block:: bash

    git clone git@github.com:metaopt/TorchOpt.git
    cd TorchOpt
    python setup.py build_from_source

The Team
--------

TorchOpt is a work by Jie Ren, Xidong Feng, Bo Liu, `Luo Mai <https://luomai.github.io/>`_ and `Yaodong Yang <https://www.yangyaodong.com/>`_.

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/metaopt/TorchOpt/issues>`_.


License
-------

TorchOpt is licensed under the Apache 2.0 License.