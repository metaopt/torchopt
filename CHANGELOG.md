# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

-

### Changed

-

### Fixed

-

### Removed

-

------

## [0.7.3] - 2023-11-10

### Changed

- Set minimal C++ standard to C++17 by [@XuehaiPan](https://github.com/XuehaiPan) in [#195](https://github.com/metaopt/torchopt/pull/195).

### Fixed

- Fix `optree` compatibility for multi-tree-map with `None` values by [@XuehaiPan](https://github.com/XuehaiPan) in [#195](https://github.com/metaopt/torchopt/pull/195).

------

## [0.7.2] - 2023-08-18

### Added

- Implement `Adadelta`, `RAdam`, `Adamax` optimizer by [@JieRen98](https://github.com/JieRen98) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#171](https://github.com/metaopt/torchopt/pull/171).

------

## [0.7.1] - 2023-05-12

### Added

- Enable CI workflow to build CXX/CUDA extension for Python 3.11 by [@XuehaiPan](https://github.com/XuehaiPan) in [#152](https://github.com/metaopt/torchopt/pull/152).
- Implement AdaGrad optimizer and exponential learning rate decay schedule by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#80](https://github.com/metaopt/torchopt/pull/80).
- Enable tests on Windows by [@XuehaiPan](https://github.com/XuehaiPan) in [#140](https://github.com/metaopt/torchopt/pull/140).
- Add `ruff` and `flake8` plugins integration by [@XuehaiPan](https://github.com/XuehaiPan) in [#138](https://github.com/metaopt/torchopt/pull/138) and [#139](https://github.com/metaopt/torchopt/pull/139).
- Add more documentation on implicit differentiation by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#143](https://github.com/metaopt/torchopt/pull/143).

### Fixed

- Fix overloaded annotations of `extract_state_dict` by [@StefanoWoerner](https://github.com/StefanoWoerner) in [#162](https://github.com/metaopt/torchopt/pull/162).
- Fix transpose empty iterable with `zip(*nested)` in transformations by [@XuehaiPan](https://github.com/XuehaiPan) in [#145](https://github.com/metaopt/torchopt/pull/145).

### Removed

- Drop Python 3.7 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#136](https://github.com/metaopt/torchopt/pull/136).

------

## [0.7.0] - 2023-02-16

### Added

- Update Sphinx documentation by [@XuehaiPan](https://github.com/XuehaiPan) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@waterhorse1](https://github.com/waterhorse1) and [@JieRen98](https://github.com/JieRen98) in [#127](https://github.com/metaopt/torchopt/pull/127).
- Add object-oriented modules support for zero-order differentiation by [@XuehaiPan](https://github.com/XuehaiPan) in [#125](https://github.com/metaopt/torchopt/pull/125).

### Changed

- Use postponed evaluation of annotations and update doctring style by [@XuehaiPan](https://github.com/XuehaiPan) in [#135](https://github.com/metaopt/torchopt/pull/135).
- Rewrite setup CUDA Toolkit logic by [@XuehaiPan](https://github.com/XuehaiPan) in [#133](https://github.com/metaopt/torchopt/pull/133).

### Fixed

- Update tests and fix corresponding bugs by [@XuehaiPan](https://github.com/XuehaiPan) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@JieRen98](https://github.com/JieRen98) in [#78](https://github.com/metaopt/torchopt/pull/78).
- Fix memory leak in implicit MAML omniglot few-shot classification example with OOP APIs by [@XuehaiPan](https://github.com/XuehaiPan) in [#113](https://github.com/metaopt/torchopt/pull/113).

------

## [0.6.0] - 2022-12-07

### Added

- Add unroll pragma for CUDA OPs by [@JieRen98](https://github.com/JieRen98) and [@XuehaiPan](https://github.com/XuehaiPan) in [#112](https://github.com/metaopt/torchopt/pull/112).
- Add Python implementation of accelerated OP and pure-Python wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#67](https://github.com/metaopt/torchopt/pull/67).
- Add `nan_to_num` hook and gradient transformation by [@XuehaiPan](https://github.com/XuehaiPan) in [#119](https://github.com/metaopt/torchopt/pull/119).
- Add matrix inversion linear solver with neumann series approximation by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#98](https://github.com/metaopt/torchopt/pull/98).
- Add if condition of number of threads for CPU OPs by [@JieRen98](https://github.com/JieRen98) in [#105](https://github.com/metaopt/torchopt/pull/105).
- Add implicit MAML omniglot few-shot classification example with OOP APIs by [@XuehaiPan](https://github.com/XuehaiPan) in [#107](https://github.com/metaopt/torchopt/pull/107).
- Add implicit MAML omniglot few-shot classification example by [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#48](https://github.com/metaopt/torchopt/pull/48).
- Add object-oriented modules support for implicit meta-gradient by [@XuehaiPan](https://github.com/XuehaiPan) in [#101](https://github.com/metaopt/torchopt/pull/101).
- Bump PyTorch version to 1.13.0 by [@XuehaiPan](https://github.com/XuehaiPan) in [#104](https://github.com/metaopt/torchopt/pull/104).
- Add zero-order gradient estimation by [@JieRen98](https://github.com/JieRen98) in [#93](https://github.com/metaopt/torchopt/pull/93).
- Add RPC-based distributed training support and add distributed MAML example by [@XuehaiPan](https://github.com/XuehaiPan) in [#83](https://github.com/metaopt/torchopt/pull/83).
- Add full type hints by [@XuehaiPan](https://github.com/XuehaiPan) in [#92](https://github.com/metaopt/torchopt/pull/92).
- Add API documentation and tutorial for implicit gradients by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@JieRen98](https://github.com/JieRen98) and [@XuehaiPan](https://github.com/XuehaiPan) in [#73](https://github.com/metaopt/torchopt/pull/73).
- Add wrapper class for functional optimizers and examples of `functorch` integration by [@vmoens](https://github.com/vmoens) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#6](https://github.com/metaopt/torchopt/pull/6).
- Implicit differentiation support by [@JieRen98](https://github.com/JieRen98) and [@waterhorse1](https://github.com/waterhorse1) and [@XuehaiPan](https://github.com/XuehaiPan) in [#41](https://github.com/metaopt/torchopt/pull/41).

### Changed

- Refactor code organization by [@XuehaiPan](https://github.com/XuehaiPan) in [#92](https://github.com/metaopt/torchopt/pull/92) and [#100](https://github/metaopt/torchopt/pull/100).

### Fixed

- Fix implicit MAML omniglot few-shot classification example by [@XuehaiPan](https://github.com/XuehaiPan) in [#108](https://github.com/metaopt/torchopt/pull/108).
- Align results of distributed examples by [@XuehaiPan](https://github.com/XuehaiPan) in [#95](https://github.com/metaopt/torchopt/pull/95).
- Fix `None` in module containers by [@XuehaiPan](https://github.com/XuehaiPan).
- Fix backward errors when using inplace `sqrt_` and `add_` by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@JieRen98](https://github.com/JieRen98) and [@XuehaiPan](https://github.com/XuehaiPan).
- Fix LR scheduling by [@XuehaiPan](https://github.com/XuehaiPan) in [#76](https://github.com/metaopt/torchopt/pull/76).
- Fix the step count tensor (`shape=(1,)`) can change the shape of the scalar updates (`shape=()`) by [@XuehaiPan](https://github.com/XuehaiPan) in [#71](https://github.com/metaopt/torchopt/pull/71).

## [0.5.0] - 2022-09-05

### Added

- Implement AdamW optimizer with masking by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#44](https://github.com/metaopt/torchopt/pull/44).
- Add half float support for accelerated OPs by [@XuehaiPan](https://github.com/XuehaiPan) in [#67](https://github.com/metaopt/torchopt/pull/67).
- Add MAML example with TorchRL integration by [@vmoens](https://github.com/vmoens) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#12](https://github.com/metaopt/TorchOpt/pull/12).
- Add optional argument `params` to update function in gradient transformations by [@XuehaiPan](https://github.com/XuehaiPan) in [#65](https://github.com/metaopt/torchopt/pull/65).
- Add option `weight_decay` option to optimizers by [@XuehaiPan](https://github.com/XuehaiPan) in [#65](https://github.com/metaopt/torchopt/pull/65).
- Add option `maximize` option to optimizers by [@XuehaiPan](https://github.com/XuehaiPan) in [#64](https://github.com/metaopt/torchopt/pull/64).
- Refactor tests using `pytest.mark.parametrize` and enabling parallel testing by [@XuehaiPan](https://github.com/XuehaiPan) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#55](https://github.com/metaopt/torchopt/pull/55).
- Add maml-omniglot few-shot classification example using functorch.vmap by [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#39](https://github.com/metaopt/torchopt/pull/39).
- Add parallel training on one GPU using functorch.vmap example by [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#32](https://github.com/metaopt/torchopt/pull/32).
- Add question/help/support issue template by [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#43](https://github.com/metaopt/torchopt/pull/43).

### Changed

- Align argument names with PyTorch by [@XuehaiPan](https://github.com/XuehaiPan) in [#65](https://github.com/metaopt/torchopt/pull/65).
- Replace JAX PyTrees with OpTree by [@XuehaiPan](https://github.com/XuehaiPan) in [#62](https://github.com/metaopt/torchopt/pull/62).
- Update image link in README to support PyPI rendering by [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#56](https://github.com/metaopt/torchopt/pull/56).

### Fixed

- Fix RMSProp optimizer by [@XuehaiPan](https://github.com/XuehaiPan) in [#55](https://github.com/metaopt/torchopt/pull/55).
- Fix momentum tracing by [@XuehaiPan](https://github.com/XuehaiPan) in [#58](https://github.com/metaopt/torchopt/pull/58).
- Fix CUDA build for accelerated OP by [@XuehaiPan](https://github.com/XuehaiPan) in [#53](https://github.com/metaopt/torchopt/pull/53).
- Fix gamma error in MAML-RL implementation by [@Benjamin-eecs](https://github.com/Benjamin-eecs) [#47](https://github.com/metaopt/torchopt/pull/47).

------

## [0.4.3] - 2022-08-08

### Added

- Bump PyTorch version to 1.12.1 by [@XuehaiPan](https://github.com/XuehaiPan) in [#49](https://github.com/metaopt/torchopt/pull/49).
- CPU-only build without `nvcc` requirement by [@XuehaiPan](https://github.com/XuehaiPan) in [#51](https://github.com/metaopt/torchopt/pull/51).
- Use [`cibuildwheel`](https://github.com/pypa/cibuildwheel) to build wheels by [@XuehaiPan](https://github.com/XuehaiPan) in [#45](https://github.com/metaopt/torchopt/pull/45).
- Use dynamic process number in CPU kernels by [@JieRen98](https://github.com/JieRen98) in [#42](https://github.com/metaopt/torchopt/pull/42).

### Changed

- Use correct Python Ctype for pybind11 function prototype [@XuehaiPan](https://github.com/XuehaiPan) in [#52](https://github.com/metaopt/torchopt/pull/52).

------

## [0.4.2] - 2022-07-26

### Added

- Read the Docs integration by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#34](https://github.com/metaopt/torchopt/pull/34).
- Update documentation and code styles by [@Benjamin-eecs](https://github.com/Benjamin-eecs) and [@XuehaiPan](https://github.com/XuehaiPan) in [#22](https://github.com/metaopt/torchopt/pull/22).
- Update tutorial notebooks by [@XuehaiPan](https://github.com/XuehaiPan) in [#27](https://github.com/metaopt/torchopt/pull/27).
- Bump PyTorch version to 1.12 by [@XuehaiPan](https://github.com/XuehaiPan) in [#25](https://github.com/metaopt/torchopt/pull/25).
- Support custom Python executable path in `CMakeLists.txt` by [@XuehaiPan](https://github.com/XuehaiPan) in [#18](https://github.com/metaopt/torchopt/pull/18).
- Add citation information by [@waterhorse1](https://github.com/waterhorse1) in [#14](https://github.com/metaopt/torchopt/pull/14) and [@Benjamin-eecs](https://github.com/Benjamin-eecs) in [#15](https://github.com/metaopt/torchopt/pull/15).
- Implement RMSProp optimizer by [@future-xy](https://github.com/future-xy) in [#8](https://github.com/metaopt/torchopt/pull/8).

### Changed

- Use `pyproject.toml` for packaging and update GitHub Action workflows by [@XuehaiPan](https://github.com/XuehaiPan) in [#31](https://github.com/metaopt/torchopt/pull/31).
- Rename the package from `TorchOpt` to `torchopt` by [@XuehaiPan](https://github.com/XuehaiPan) in [#20](https://github.com/metaopt/torchopt/pull/20).

### Fixed

- Fixed errors while building from the source and add `conda` environment recipe by [@XuehaiPan](https://github.com/XuehaiPan) in [#24](https://github.com/metaopt/torchopt/pull/24).

------

## [0.4.1] - 2022-04-15

### Fixed

- Fix set devices bug for multi-GPUs.

------

## [0.4.0] - 2022-04-09

### Added

- The first beta release of TorchOpt.
- TorchOpt with L2R, LOLA, MAML-RL, MGRL, and few-shot examples.

------

[Unreleased]: https://github.com/metaopt/torchopt/compare/v0.7.3...HEAD
[0.7.3]: https://github.com/metaopt/torchopt/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/metaopt/torchopt/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/metaopt/torchopt/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/metaopt/torchopt/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/metaopt/torchopt/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/metaopt/torchopt/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/metaopt/torchopt/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/metaopt/torchopt/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/metaopt/torchopt/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/metaopt/torchopt/releases/tag/v0.4.0
