__version__ = "0.0.0"

from ._src.alias import sgd, adam, custom_adam, meta_sgd, meta_adam
from ._src.update import apply_updates

from ._src.Optimizer import Optimizer, SGD, Adam, CustomAdam
from ._src.MetaOptimizer import MetaOptimizer, MetaSGD, MetaAdam, CustomMetaAdam

from ._src.utils import stop_gradient, extract_state_dict, recover_state_dict

from ._src import hook
from ._src import visual
from ._src import clip
from ._src import combine
