print-%  : ; @echo $* = $($*)
PROJECT_NAME   = torchopt
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) examples include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -name "*.h" -o -name "*.cpp")
CUDA_FILES     = $(shell find $(SOURCE_FOLDERS) -type f -name "*.cuh" -o -name "*.cu")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
CLANG_FORMAT   ?= $(shell command -v clang-format-14 || command -v clang-format)
PYTESTOPTS     ?=

.PHONY: default
default: install

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install torch numpy pybind11
	USE_FP16=ON TORCH_CUDA_ARCH_LIST=Auto $(PYTHON) -m pip install -vvv --no-build-isolation --editable .

install-e: install-editable  # alias

uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(1) --upgrade)
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(2) --upgrade)

pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install,flake8-bugbear)
	$(call check_pip_install,flake8-comprehensions)
	$(call check_pip_install,flake8-docstrings)
	$(call check_pip_install,flake8-pyi)
	$(call check_pip_install,flake8-simplify)

py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install_extra,black,black[jupyter])

ruff-install:
	$(call check_pip_install,ruff)

mypy-install:
	$(call check_pip_install,mypy)

pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

docs-install:
	$(call check_pip_install_extra,pydocstyle,pydocstyle[toml])
	$(call check_pip_install,doc8)
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-rtd-theme)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinxcontrib-katex)
	$(call check_pip_install,sphinxcontrib-bibtex)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install,myst-nb)
	$(call check_pip_install_extra,sphinxcontrib-spelling,sphinxcontrib-spelling pyenchant)

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

test-install: pytest-install
	$(PYTHON) -m pip install --requirement tests/requirements.txt

cmake-install:
	command -v cmake || $(call check_pip_install,cmake)

cpplint-install:
	$(call check_pip_install,cpplint)

clang-format-install:
	command -v clang-format-14 || command -v clang-format || \
	sudo apt-get install -y clang-format-14 || \
	sudo apt-get install -y clang-format

clang-tidy-install:
	command -v clang-tidy || sudo apt-get install -y clang-tidy

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

pytest: test-install
	cd tests && $(PYTHON) -c 'import $(PROJECT_NAME)' && \
	$(PYTHON) -m pytest -k "test_adagrad" --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_NAME)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

test: pytest

# Python linters

pylint: pylint-install
	$(PYTHON) -m pylint $(PROJECT_PATH)

flake8: flake8-install
	$(PYTHON) -m flake8 --count --show-source --statistics

py-format: py-format-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES) tutorials

ruff: ruff-install
	$(PYTHON) -m ruff check .

ruff-fix: ruff-install
	$(PYTHON) -m ruff check . --fix --exit-non-zero-on-fix

mypy: mypy-install
	$(PYTHON) -m mypy $(PROJECT_PATH)

pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit run --all-files

# C++ linters

cmake-configure: cmake-install
	cmake -S . -B cmake-build-debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPYTHON_EXECUTABLE="$(PYTHON)"

cmake-build: cmake-configure
	cmake --build cmake-build-debug --parallel

cmake: cmake-build

cpplint: cpplint-install
	$(PYTHON) -m cpplint $(CXX_FILES) $(CUDA_FILES)

clang-format: clang-format-install
	$(CLANG_FORMAT) --style=file -i $(CXX_FILES) $(CUDA_FILES) -n --Werror

clang-tidy: clang-tidy-install cmake-configure
	clang-tidy -p=cmake-build-debug $(CXX_FILES)

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") -check $(SOURCE_FOLDERS)

docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

clean-docs:
	make -C docs clean

# Utility functions

lint: ruff flake8 py-format mypy pylint clang-format clang-tidy cpplint addlicense docstyle spelling

format: py-format-install ruff-install clang-format-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES) tutorials
	$(PYTHON) -m ruff check . --fix --exit-zero
	$(CLANG_FORMAT) -style=file -i $(CXX_FILES) $(CUDA_FILES)
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm tests/.coverage
	rm tests/coverage.xml

clean-build:
	rm -rf build/ dist/
	rm -rf *.egg-info .eggs

clean: clean-py clean-build clean-docs

# Build docker images

docker-base:
	docker build --target base --tag $(PROJECT_NAME):$(COMMIT_HASH) --file Dockerfile .
	@echo Successfully build docker image with tag $(PROJECT_NAME):$(COMMIT_HASH)

docker-devel:
	docker build --target devel --tag $(PROJECT_NAME)-devel:$(COMMIT_HASH) --file Dockerfile .
	@echo Successfully build docker image with tag $(PROJECT_NAME)-devel:$(COMMIT_HASH)

docker: docker-base docker-devel

docker-run-devel: docker-devel
	docker run --network=host --gpus=all -v /:/host -h ubuntu -it $(PROJECT_NAME)-devel:$(COMMIT_HASH)
