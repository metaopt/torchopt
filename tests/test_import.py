# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torchopt


def test_accelerated_op_import() -> None:
    torchopt.accelerated_op.adam_op.AdamOp
    torchopt.accelerated_op.is_available
    torchopt.accelerated_op_available
    from torchopt.accelerated_op import is_available
    from torchopt.accelerated_op.adam_op import AdamOp


def test_alias_import() -> None:
    torchopt.adadelta
    torchopt.adagrad
    torchopt.adam
    torchopt.adamw
    torchopt.adamax
    torchopt.radam
    torchopt.rmsprop
    torchopt.sgd
    torchopt.alias.adadelta
    torchopt.alias.adagrad
    torchopt.alias.adam
    torchopt.alias.adamw
    torchopt.alias.adamax
    torchopt.alias.radam
    torchopt.alias.rmsprop
    torchopt.alias.sgd
    from torchopt import adadelta, adagrad, adam, adamax, adamw, radam, rmsprop, sgd
    from torchopt.alias import adadelta, adagrad, adam, adamax, adamw, radam, rmsprop, sgd


def test_diff_import() -> None:
    torchopt.diff.implicit
    torchopt.diff.implicit.custom_root
    torchopt.diff.implicit.ImplicitMetaGradientModule
    torchopt.diff.implicit.nn.ImplicitMetaGradientModule
    torchopt.diff.zero_order
    torchopt.diff.zero_order.zero_order
    torchopt.diff.zero_order.ZeroOrderGradientModule
    torchopt.diff.zero_order.nn.ZeroOrderGradientModule
    from torchopt.diff import implicit, zero_order
    from torchopt.diff.implicit import ImplicitMetaGradientModule, custom_root
    from torchopt.diff.zero_order import ZeroOrderGradientModule, zero_order


def test_distributed_import() -> None:
    torchopt.distributed.api
    torchopt.distributed.autograd
    torchopt.distributed.world
    torchopt.distributed.is_available
    torchopt.distributed.TensorDimensionPartitioner
    torchopt.distributed.dim_partitioner
    torchopt.distributed.batch_partitioner
    torchopt.distributed.mean_reducer
    torchopt.distributed.sum_reducer
    torchopt.distributed.remote_async_call
    torchopt.distributed.remote_sync_call
    torchopt.distributed.parallelize
    torchopt.distributed.parallelize_async
    torchopt.distributed.parallelize_sync
    torchopt.distributed.get_world_info
    torchopt.distributed.get_world_rank
    torchopt.distributed.get_rank
    torchopt.distributed.get_world_size
    torchopt.distributed.get_local_rank
    torchopt.distributed.get_local_world_size
    torchopt.distributed.get_worker_id
    torchopt.distributed.barrier
    torchopt.distributed.auto_init_rpc
    torchopt.distributed.on_rank
    torchopt.distributed.not_on_rank
    torchopt.distributed.rank_zero_only
    torchopt.distributed.rank_non_zero_only
    torchopt.distributed.autograd.is_available
    torchopt.distributed.autograd.context
    from torchopt.distributed import api, autograd, world


def test_linalg_import() -> None:
    torchopt.linalg.cg
    torchopt.linalg.ns
    torchopt.linalg.ns_inv
    from torchopt.linalg import cg, ns, ns_inv


def test_linear_solve_import() -> None:
    torchopt.linear_solve.solve_cg
    torchopt.linear_solve.solve_inv
    torchopt.linear_solve.solve_normal_cg
    from torchopt.linear_solve import solve_cg, solve_inv, solve_normal_cg


def test_nn_import() -> None:
    torchopt.nn.MetaGradientModule
    torchopt.nn.ImplicitMetaGradientModule
    torchopt.nn.ZeroOrderGradientModule
    from torchopt.nn import ImplicitMetaGradientModule, MetaGradientModule, ZeroOrderGradientModule


def test_optim_import() -> None:
    torchopt.FuncOptimizer
    torchopt.MetaAdaDelta
    torchopt.MetaAdadelta
    torchopt.MetaAdaGrad
    torchopt.MetaAdagrad
    torchopt.MetaAdam
    torchopt.MetaAdamW
    torchopt.MetaAdaMax
    torchopt.MetaAdamax
    torchopt.MetaRAdam
    torchopt.MetaRMSProp
    torchopt.MetaRMSprop
    torchopt.MetaSGD
    torchopt.AdaDelta
    torchopt.Adadelta
    torchopt.AdaGrad
    torchopt.Adagrad
    torchopt.Adam
    torchopt.AdamW
    torchopt.AdaMax
    torchopt.Adamax
    torchopt.Optimizer
    torchopt.RMSProp
    torchopt.RMSprop
    torchopt.SGD
    torchopt.optim.meta.MetaAdaDelta
    torchopt.optim.meta.MetaAdadelta
    torchopt.optim.meta.MetaAdaGrad
    torchopt.optim.meta.MetaAdagrad
    torchopt.optim.meta.MetaAdam
    torchopt.optim.meta.MetaAdamW
    torchopt.optim.meta.MetaAdaMax
    torchopt.optim.meta.MetaAdamax
    torchopt.optim.meta.MetaRMSProp
    torchopt.optim.meta.MetaRMSprop
    torchopt.optim.meta.MetaSGD
    torchopt.optim.Adam
    torchopt.optim.AdamW
    torchopt.optim.Optimizer
    torchopt.optim.RMSProp
    torchopt.optim.RMSprop
    torchopt.optim.SGD
    torchopt.optim.func.FuncOptimizer
    from torchopt import (
        SGD,
        AdaDelta,
        Adadelta,
        AdaGrad,
        Adagrad,
        Adam,
        AdaMax,
        Adamax,
        AdamW,
        FuncOptimizer,
        MetaAdaDelta,
        MetaAdadelta,
        MetaAdaGrad,
        MetaAdagrad,
        MetaAdam,
        MetaAdaMax,
        MetaAdamax,
        MetaAdamW,
        MetaOptimizer,
        MetaRMSprop,
        MetaRMSProp,
        MetaSGD,
        Optimizer,
        RMSProp,
    )
    from torchopt.optim import SGD, Adam, AdamW, FuncOptimizer, Optimizer, RMSProp
    from torchopt.optim.func import FuncOptimizer
    from torchopt.optim.meta import (
        MetaAdaDelta,
        MetaAdadelta,
        MetaAdaGrad,
        MetaAdagrad,
        MetaAdam,
        MetaAdaMax,
        MetaAdamax,
        MetaAdamW,
        MetaOptimizer,
        MetaRAdam,
        MetaRMSProp,
        MetaRMSprop,
        MetaSGD,
    )


def test_schedule_import() -> None:
    torchopt.schedule.linear_schedule
    torchopt.schedule.polynomial_schedule
    from torchopt.schedule import linear_schedule, polynomial_schedule


def test_transform_import() -> None:
    torchopt.transform.add_decayed_weights
    torchopt.transform.scale
    torchopt.transform.scale_by_accelerated_adam
    torchopt.transform.scale_by_adam
    torchopt.transform.scale_by_rms
    torchopt.transform.scale_by_schedule
    torchopt.transform.scale_by_stddev
    torchopt.transform.trace
    torchopt.transform.nan_to_num
    torchopt.nan_to_num
    from torchopt import nan_to_num
    from torchopt.transform import (
        add_decayed_weights,
        nan_to_num,
        scale,
        scale_by_accelerated_adam,
        scale_by_adam,
        scale_by_rms,
        scale_by_schedule,
        scale_by_stddev,
        trace,
    )


def test_base_import() -> None:
    torchopt.base.EmptyState
    torchopt.base.GradientTransformation
    torchopt.base.ChainedGradientTransformation
    torchopt.base.identity
    from torchopt.base import (
        ChainedGradientTransformation,
        EmptyState,
        GradientTransformation,
        identity,
    )


def test_clip_import() -> None:
    torchopt.clip_grad_norm
    torchopt.clip.clip_grad_norm
    from torchopt import clip_grad_norm
    from torchopt.clip import clip_grad_norm


def test_combine_import() -> None:
    torchopt.chain
    torchopt.chain.flat
    torchopt.combine.chain
    torchopt.combine.chain.flat
    torchopt.combine.chain_flat
    from torchopt import chain
    from torchopt.combine import chain, chain_flat


def test_hook_import() -> None:
    torchopt.register_hook
    torchopt.hook.register_hook
    torchopt.hook.zero_nan_hook
    torchopt.hook.nan_to_num_hook
    from torchopt import register_hook
    from torchopt.hook import nan_to_num_hook, register_hook, zero_nan_hook


def test_pytree_import() -> None:
    torchopt.pytree.tree_flatten_as_tuple
    torchopt.pytree.tree_pos
    torchopt.pytree.tree_neg
    torchopt.pytree.tree_add
    torchopt.pytree.tree_add_scalar_mul
    torchopt.pytree.tree_sub
    torchopt.pytree.tree_sub_scalar_mul
    torchopt.pytree.tree_mul
    torchopt.pytree.tree_matmul
    torchopt.pytree.tree_scalar_mul
    torchopt.pytree.tree_truediv
    torchopt.pytree.tree_vdot_real
    torchopt.pytree.tree_wait
    from torchopt.pytree import (
        tree_add,
        tree_add_scalar_mul,
        tree_flatten_as_tuple,
        tree_matmul,
        tree_mul,
        tree_neg,
        tree_pos,
        tree_scalar_mul,
        tree_sub,
        tree_sub_scalar_mul,
        tree_truediv,
        tree_vdot_real,
        tree_wait,
    )


def test_typing_import() -> None:
    torchopt.typing.GradientTransformation
    torchopt.typing.ChainedGradientTransformation
    torchopt.typing.EmptyState
    torchopt.typing.UninitializedState
    torchopt.typing.Params
    torchopt.typing.Updates
    torchopt.typing.OptState
    torchopt.typing.Scalar
    torchopt.typing.Numeric
    torchopt.typing.Schedule
    torchopt.typing.ScalarOrSchedule
    torchopt.typing.PyTree
    torchopt.typing.Tensor
    torchopt.typing.OptionalTensor
    torchopt.typing.ListOfTensors
    torchopt.typing.TupleOfTensors
    torchopt.typing.SequenceOfTensors
    torchopt.typing.TensorOrTensors
    torchopt.typing.TensorTree
    torchopt.typing.ListOfOptionalTensors
    torchopt.typing.TupleOfOptionalTensors
    torchopt.typing.SequenceOfOptionalTensors
    torchopt.typing.OptionalTensorOrOptionalTensors
    torchopt.typing.OptionalTensorTree
    torchopt.typing.TensorContainer
    torchopt.typing.ModuleTensorContainers
    torchopt.typing.Future
    torchopt.typing.LinearSolver
    torchopt.typing.Device
    torchopt.typing.Size
    torchopt.typing.Distribution
    torchopt.typing.SampleFunc
    torchopt.typing.Samplable
    from torchopt.typing import (
        ChainedGradientTransformation,
        Device,
        Distribution,
        EmptyState,
        Future,
        GradientTransformation,
        LinearSolver,
        ListOfOptionalTensors,
        ListOfTensors,
        ModuleTensorContainers,
        Numeric,
        OptionalTensor,
        OptionalTensorOrOptionalTensors,
        OptionalTensorTree,
        OptState,
        Params,
        PyTree,
        Samplable,
        SampleFunc,
        Scalar,
        ScalarOrSchedule,
        Schedule,
        SequenceOfOptionalTensors,
        SequenceOfTensors,
        Size,
        Tensor,
        TensorContainer,
        TensorOrTensors,
        TensorTree,
        TupleOfOptionalTensors,
        TupleOfTensors,
        UninitializedState,
        Updates,
    )


def test_update_import() -> None:
    torchopt.apply_updates
    torchopt.update.apply_updates
    from torchopt import apply_updates
    from torchopt.update import apply_updates


def test_utils_import() -> None:
    torchopt.utils.ModuleState
    torchopt.utils.stop_gradient
    torchopt.utils.extract_state_dict
    torchopt.utils.recover_state_dict
    torchopt.utils.module_clone
    torchopt.utils.module_detach_
    from torchopt.utils import (
        ModuleState,
        extract_state_dict,
        module_clone,
        module_detach_,
        recover_state_dict,
        stop_gradient,
    )


def test_version_import() -> None:
    torchopt.__version__
    torchopt.version.__version__
    from torchopt import __version__
    from torchopt.version import __version__


def test_visual_import() -> None:
    torchopt.visual.make_dot
    torchopt.visual.resize_graph
    from torchopt.visual import make_dot, resize_graph
