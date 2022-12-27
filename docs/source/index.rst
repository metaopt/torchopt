:github_url: https://github.com/metaopt/torchopt/tree/HEAD/docs

TorchOpt
--------

**TorchOpt** is an efficient library for differentiable optimization built upon `PyTorch <https://pytorch.org>`_.
Torchopt is

* **Comprehensive**: TorchOpt provides three differentiation mode - explicit differentiation, implicit differentiation and zero-order differentiation for handling different differentiable optimization situations.
* **Flexible**: TorchOpt provides both functional and objective-oriented API for user different preferences. Users can implement differentiable optimization in JAX-like or PyTorch-like style.
* **Efficient**: TorchOpt provides (1) CPU/GPU acceleration differentiable optimizer (2) RPC-based distributed training framework (3) Fast Tree Operations, to largely increase the training efficiency for bi-level optimization problem.

Installation
------------

Requirements:

* `PyTorch <https://pytorch.org>`_
* (Optional) `Graphviz <https://graphviz.org/download>`_

Please follow the instructions at https://pytorch.org to install PyTorch in your Python environment first. Then run the following command to install TorchOpt from PyPI:

.. code-block:: bash

    pip install torchopt

You can also build shared libraries from source, use:

.. code-block:: bash

    git clone https://github.com/metaopt/torchopt.git
    cd torchopt
    pip3 install .

We provide a `conda <https://github.com/conda/conda>`_ environment recipe to install the build toolchain such as `cmake`, `g++`, and `nvcc`:

.. code-block:: bash

    git clone https://github.com/metaopt/torchopt.git
    cd torchopt

    # You may need `CONDA_OVERRIDE_CUDA` if conda fails to detect the NVIDIA driver (e.g. in docker or WSL2)
    CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda-recipe-minimal.yaml

    conda activate torchopt


.. toctree::
    :caption: Getting Started
    :maxdepth: 1

    torchopt101/torchopt-101.rst


.. toctree::
    :caption: Examples
    :maxdepth: 1

    examples/MAML.rst


.. toctree::
    :caption: Developer Documentation
    :maxdepth: 1

    developer/contributing.rst
    developer/contributor.rst

.. toctree::
    :caption: API Documentation
    :maxdepth: 2

    api/api.rst

The Team
--------

TorchOpt is a work by

* Jie Ren (`JieRen98 <https://github.com/JieRen98>`_)
* Xidong Feng (`waterhorse1 <https://github.com/waterhorse1>`_)
* Bo Liu (`Benjamin-eecs <https://github.com/Benjamin-eecs>`_)
* Xuehai Pan (`XuehaiPan <https://github.com/XuehaiPan>`_)
* Luo Mai (`luomai <https://luomai.github.io/>`_)
* Yaodong Yang (`PKU-YYang <https://www.yangyaodong.com/>`_).

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/metaopt/torchopt/issues>`_.

Changelog
---------

See :gitcode:`CHANGELOG.md`.

License
-------

TorchOpt is licensed under the Apache 2.0 License.

Citing
------

If you find TorchOpt useful, please cite it in your publications.

.. code-block:: bibtex

    @article{torchopt,
      title   = {TorchOpt: An Efficient Library for Differentiable Optimization},
      author  = {Ren, Jie and Feng, Xidong and Liu, Bo and Pan, Xuehai and Fu, Yao and Mai, Luo and Yang, Yaodong},
      journal = {arXiv preprint arXiv:2211.06934},
      year    = {2022}
    }


Indices and tables
==================

* :ref:`genindex`
