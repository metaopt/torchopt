Distributed Training
====================

Distributed training is a technique that allows you to train your pipeline on multiple worker/machines.
This is useful when you have a large model or computation graph that doesn't fit on a single GPU/machine, or when you want to train a model faster by using more resources.

TorchOpt offers a simple API to train your model on multiple GPUs/machines based on the PyTorch |Distributed RPC|_.
Here are some key concepts that TorchOpt's distributed mechanism relies on:

- **Remote Procedure Call (RPC)** supports running a function on the specified destination worker with the given arguments and getting the return value back or creating a reference to the return value.

  That is, you can treat the remote worker as a accelerator. You can call a function on a remote worker and get the result back to the local worker.

- **Distributed Autograd** stitches together local autograd engines on all the workers involved in the forward pass, and automatically reach out to them during the backward pass to compute gradients.

  This is much more flexible to fit the meta-learning use case to have a complex task dependency tree.

.. |Distributed RPC| replace:: Distributed RPC Framework (``torch.distributed.rpc``)
.. _Distributed RPC: https://pytorch.org/docs/stable/rpc.html

Here are some useful resources to learn more about distributed training:

- `Distributed RPC Framework <https://pytorch.org/docs/stable/rpc.html>`_
- `Distributed Autograd Design <https://pytorch.org/docs/stable/rpc/distributed_autograd.html>`_
- `Remote Reference Protocol <https://pytorch.org/docs/stable/rpc/rref.html#remote-reference-protocol>`_
- `RPC tutorials <https://pytorch.org/docs/stable/rpc.html#tutorials>`_
- `Autograd mechanics <https://pytorch.org/docs/stable/notes/autograd.html>`_

------

Why RPC-Based Distributed Training
----------------------------------

Due to the Global Interpreter Lock (GIL) in Python, only one thread can execute Python code at a time.
This means that you can't take advantage of multiple cores on your machine.
Distribute the workload cross multiple processes, or namely workers, that will run in parallel to gain faster execution performance.
Each worker will have its own Python interpreter and memory namespace.

Compare to single-process programming, you need to be aware of the following:

- **Communication**: You need to explicitly send and receive messages between workers.
- **Synchronization**: You need to explicitly synchronize the states between workers.

Message Passing Interface (MPI) and Distributed Data-Parallel Training (DDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`MPI <https://www.mpi-forum.org>`_ is a standard for message passing between processes.
It is a popular choice for `Distributed Data-Parallel Training (DDP) <https://pytorch.org/tutorials/beginner/dist_overview.html>`_.
PyTorch has implemented this with several `backends <https://pytorch.org/docs/stable/distributed.html#backends>`_, including `Gloo <https://github.com/facebookincubator/gloo>`_, `MPI <https://www.mpi-forum.org>`_, and `NCCL <https://developer.nvidia.com/nccl>`_.

However, MPI-based parallelism has some drawbacks:

- **MPI is not user-friendly**.
  MPI-like APIs only provides low-level primitives for sending and receiving messages.
  It requires the users manage the message passing between workers manually.
  The users should be aware of the communication pattern and the synchronization between workers.

- **MPI is not flexible**.
  MPI-like APIs are designed for `Distributed Data-Parallel Training (DDP) <https://pytorch.org/tutorials/beginner/dist_overview.html>`_, which is a widely adopted `single-program multiple-data (SPMD) <https://en.wikipedia.org/wiki/Single_program,_multiple_data>`_ training paradigm.
  However, for meta-learning tasks, the task dependency tree is complex and dynamic.
  It may not fit into the SPMD paradigm.
  It is hard to implement the distributed Autograd engine on top of MPI.

- **MPI only communicates the value of tensors and not the gradients**.
  This is a limitation of MPI.
  The users need to handle the gradients manually cross multiple workers.
  For example, receive the gradients from other workers and put them as ``grad_outputs`` to function |torch.autograd.grad|_.

.. |torch.autograd.grad| replace:: ``torch.autograd.grad``
.. _torch.autograd.grad: https://pytorch.org/docs/stable/generated/torch.autograd.grad.html

Remote Procedure Call (RPC) and Distributed Autograd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To address the needs of meta-learning tasks, which have complex and dynamic nature of the training process.
TorchOpt uses PyTorch |Distributed RPC|_ to implement the distributed training mechanism.
PyTorch implements the RPC communication operations with appropriate ``rpcSendBackward`` and ``rpcRecvBackward`` functions.

With **RPC** and **Distributed Autograd**, TorchOpt distributes a **differentiable optimization** job across multiple workers and executes the workers in parallel.
It allows the users to build the whole computation graph (both forward an backward) cross multiple workers.
The users can wrap code in the distributed Autograd module and achieve substantial speedup in training time with only a few changes in existing training scripts.

Here is an example of distributed autograd graph using RPC:

.. image:: https://pytorch.org/docs/stable/_images/distributed_dependencies_computed.png

For more details, please refer to the `Distributed Autograd Design <https://pytorch.org/docs/stable/rpc/distributed_autograd.html>`_ documentation.

------

TorchOpt's Distributed Training
-------------------------------

TorchOpt's distributed package is built upon the PyTorch |Distributed RPC|_ and |Distributed Autograd Framework|_.

.. |Distributed Autograd Framework| replace:: Distributed Autograd Framework (``torch.distributed.autograd``)
.. _Distributed Autograd Framework: https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework

TorchOpt provides some utility functions to make it easier to use the distributed training mechanism.

Initialization and Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.auto_init_rpc
    torchopt.distributed.barrier

Users can wrap their program entry function with decorator :func:`torchopt.distributed.auto_init_rpc`:

.. code-block:: python

    import torchopt.distributed as todist

    def parse_arguments():
        parser = argparse.ArgumentParser()
        ...

        return args

    def worker_init_fn():
        # set process title, seeding, etc.

    @todist.auto_init_rpc(worker_init_fn)
    def main():
        # Your code here
        args = parse_arguments()
        ...

    if __name__ == '__main__':
        main()

The decorator will initialize the RPC framework and synchronize the workers on startup.

.. note::

    By default, all tensors must move to the CPU before sending them to other workers.
    If you want to send/receive the tensors directly between GPUs from different workers, you need to specify the ``rpc_backend_options`` with ``device_maps``.
    Please refer to the documentation of |torch.distributed.rpc.init_rpc|_ for more details.

.. |torch.distributed.rpc.init_rpc| replace:: ``torch.distributed.rpc.init_rpc``
.. _torch.distributed.rpc.init_rpc: https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc

Then, users can use |torchrun|_ to launch the program:

.. code-block:: bash

    torchrun --nnodes=1 --nproc_per_node=8 YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

.. |torchrun| replace:: ``torchrun`` (Elastic Launch)
.. _torchrun: https://pytorch.org/docs/stable/elastic/run.html

Process group information
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.get_world_info
    torchopt.distributed.get_world_rank
    torchopt.distributed.get_rank
    torchopt.distributed.get_world_size
    torchopt.distributed.get_local_rank
    torchopt.distributed.get_local_world_size
    torchopt.distributed.get_worker_id

After initialize the RPC server, users can use the above functions to get the process group information.

For example, use :func:`torchopt.distributed.get_local_rank` to determine which GPU device to use:

.. code-block:: python

    import torch
    import torchopt.distributed as todist

    def worker_init_fn():
        local_rank = todist.get_local_rank()
        torch.cuda.set_device(local_rank)

    @todist.auto_init_rpc(worker_init_fn)
    def main():
        ...

Worker selection
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.on_rank
    torchopt.distributed.not_on_rank
    torchopt.distributed.rank_zero_only
    torchopt.distributed.rank_non_zero_only

TorchOpt provides some decorators to execute the decorated function on specific workers.

For example, use :func:`torchopt.distributed.rank_zero_only` to execute the function only on the main worker (``worker0``), such as saving checkpoints or logging the results:

.. code-block:: python

    import torchopt.distributed as todist

    @todist.rank_zero_only
    def save_checkpoint(model):
        ...

    @todist.rank_zero_only
    def log_results(writer, results):
        ...

    @todist.auto_init_rpc(worker_init_fn)
    def main():
        ...

        for epoch in range(args.epochs):
            ...

            if epoch % args.log_interval == 0:
                log_results(writer, results)

            if epoch % args.save_interval == 0:
                save_checkpoint(model)

Remote Procedure Call (RPC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.remote_async_call
    torchopt.distributed.remote_sync_call

TorchOpt provides two functions to execute the remote procedure call (RPC) on the remote workers.
The asynchronized version :func:`remote_async_call` function returns a |torch.Future|_ object, and the :func:`remote_sync_call` function executes and returns the result directly.

.. |torch.Future| replace:: ``torch.Future``
.. _torch.Future: https://pytorch.org/docs/stable/futures.html#torch.futures.Future

Users can distribute their workload (a function) to a specific worker by:

.. code-block:: python

    import torchopt.distributed as todist

    @todist.auto_init_rpc(worker_init_fn)
    def main():
        ...

        # Execute the function on the remote worker (asynchronously)
        future = todist.remote_async_call(func, args=(arg1, arg2, ...), kwargs={...}, partitioner=worker_id)

        # Wait for the result
        result = future.wait()

        ...

or

.. code-block:: python

    import torchopt.distributed as todist

    @todist.auto_init_rpc(worker_init_fn)
    def main():
        ...

        # Execute the function on the remote worker
        result = todist.remote_sync_call(func, args=(arg1, arg2, ...), kwargs={...}, partitioner=worker_id)

        ...

TorchOpt follows the `MapReduce programming model <https://en.wikipedia.org/wiki/MapReduce>`_ to distribute the workload.

The ``partitioner`` argument specifies the worker to execute the function.
The users can optionally specify the ``reducer`` argument to aggregate the results from the workers.
Finally, the caller will get the reference to the result on the local worker.

- ``partitioner``: a function that takes the ``args`` and ``kwargs`` arguments and returns a list of triplets ``(worker_id, worker_args, worker_kwargs)``.

  The ``partitioner`` is responsible for partitioning the workload (inputs) and distributing them to the remote workers.

  If the ``partitioner`` is given by a worker ID (:class:`int` or :class:`str`), the function will be executed on the specified worker.

- ``mapper``: the ``func`` argument to be executed on the remote worker.
- ``reducer`` (optional): aggregation function, takes a list of results from the remote workers and returns the final result.

  If the ``reducer`` is not given, returns the original unaggregated list.

Predefined partitioners and reducers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.dim_partitioner
    torchopt.distributed.batch_partitioner
    torchopt.distributed.mean_reducer
    torchopt.distributed.sum_reducer

We provide some predefined partitioners and reducers.
Users can combine the :func:`torchopt.distributed.batch_partitioner` and :func:`torchopt.distributed.mean_reducer` to achieve the distributed data parallelism (DDP) easily:

.. code-block:: python

    import torchopt.distributed as todist

    def loss_fn(model, batch):
        ...

    @todist.rank_zero_only
    def train(args):

        for epoch in range(args.epochs):
            ...

            for batch in dataloader:
                # Partition the data on the batch (first) dimension and distribute them to the remote workers
                # Aggregate the results from the remote workers and return the mean loss
                loss = todist.remote_sync_call(
                    loss_fn,
                    args=(model, batch),
                    partitioner=todist.batch_partitioner,
                    reducer=todist.mean_reducer,
                )

                ...

We also provide a :func:`torchopt.distributed.dim_partitioner` to partition the data on the specified dimension.
While implementing the **Model-Agnostic Meta-Learning** (MAML) :cite:`MAML` algorithm, users can use this to parallel the training for the inner loop:

.. code-block:: python

    import torchopt.distributed as todist

    def inner_loop(model, task_batch, args):
        # task_batch: shape = (B, *)
        inner_model = torchopt.module_clone(model, by='reference', detach_buffers=True)

        # Inner optimization
        for inner_step in range(args.inner_steps):
            inner_loss = inner_loss_fn(inner_model, task_batch)

            # Update the inner model
            ...

        # Compute the outer loss
        outer_loss = inner_loss_fn(inner_model, task_batch)
        return outer_loss

    @todist.rank_zero_only
    def train(args):

        for epoch in range(args.epochs):
            ...

            for batch in dataloader:
                # batch: shape = (T, B, *)
                outer_loss = todist.remote_sync_call(
                    inner_loop,
                    args=(model, batch),
                    partitioner=todist.dim_partitioner(0, exclusive=True, keepdim=False),
                    reducer=todist.mean_reducer,
                )

                ...

The the ``dim_partitioner(0, exclusive=True, keepdim=False)`` will split the batch of size ``(T, B, *)`` into ``T`` batches of size ``(B, *)``.
Each task will be executed on the remote worker independently.
Finally, the results will be aggregated by the :func:`torchopt.distributed.mean_reducer` to compute the mean loss.
Inside the ``inner_loop`` function, users may use another RPC call to further parallelize the inner loop.

Function parallelization wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.parallelize
    torchopt.distributed.parallelize_async
    torchopt.distributed.parallelize_sync

TorchOpt offers wrappers to parallelize the function execution on the remote workers.
It makes the function execution on the remote workers more transparent to the users and makes the code structure more clear.

.. code-block:: python

    import torchopt.distributed as todist

    @todist.parallelize(partitioner=todist.batch_partitioner, reducer=todist.mean_reducer)
    def distributed_data_parallelism(model, batch, args):
        # Compute local loss of the given batch
        ...
        return loss

    @todist.parallelize(
        partitioner=todist.dim_partitioner(0, exclusive=True, keepdim=False),
        reducer=todist.mean_reducer,
    )
    def inner_loop(model, batch, args):  # distributed MAML inner loop
        # batch: shape = (B, *)
        inner_model = torchopt.module_clone(model, by='reference', detach_buffers=True)

        # Inner optimization
        ...

        # Compute the outer loss
        outer_loss = inner_loss_fn(inner_model, task_batch)
        return outer_loss

    @todist.rank_zero_only
    def train(args):

        for epoch in range(args.epochs):
            ...

            for batch in dataloader:
                # batch: shape = (T, B, *)
                outer_loss = inner_loop(model, batch, args)

                ...

Distributed Autograd
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../_autosummary

    torchopt.distributed.autograd.context
    torchopt.distributed.autograd.get_gradients
    torchopt.distributed.autograd.backward
    torchopt.distributed.autograd.grad
