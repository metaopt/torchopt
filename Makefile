print-%  : ; @echo $* = $($*)
PROJECT_NAME   = torchopt
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) examples include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -name "*.h" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.cu")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
CLANG_FORMAT   ?= $(shell command -v clang-format-14 || command -v clang-format)

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
	$(call check_pip_install,pylint)

flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install_extra,flake8-bugbear,flake8-bugbear)

py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install_extra,black,black[jupyter])

mypy-install:
	$(call check_pip_install,mypy)

pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

docs-install:
	$(call check_pip_install,pydocstyle)
	$(call check_pip_install_extra,doc8,"doc8<1.0.0a0")
	if ! $(PYTHON) -c "import sys; exit(sys.version_info < (3, 8))"; then \
		$(PYTHON) -m pip uninstall --yes importlib-metadata; \
		$(call check_pip_install_extra,importlib-metadata,"importlib-metadata<5.0.0a0"); \
	fi
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-rtd-theme)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinxcontrib-katex)
	$(call check_pip_install,sphinxcontrib-bibtex)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install,myst_nb)
	$(call check_pip_install_extra,sphinxcontrib.spelling,sphinxcontrib.spelling pyenchant)

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest_cov)
	$(call check_pip_install,pytest_xdist)

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
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

pytest: pytest-install
	cd tests && \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_NAME)" --cov-report=xml --cov-report=term-missing \
		.

test: pytest

# Python linters

pylint: pylint-install
	$(PYTHON) -m pylint $(PROJECT_PATH)

flake8: flake8-install
	$(PYTHON) -m flake8 $(PYTHON_FILES) --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

py-format: py-format-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES) tutorials

mypy: mypy-install
	$(PYTHON) -m mypy $(PROJECT_PATH)

pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit run --all-files

# C++ linters

cpplint: cpplint-install
	$(PYTHON) -m cpplint $(CXX_FILES)

clang-format: clang-format-install
	$(CLANG_FORMAT) --style=file -i $(CXX_FILES) -n --Werror

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2022 -check $(SOURCE_FOLDERS)

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

lint: flake8 py-format mypy pylint clang-format cpplint docstyle spelling

format: py-format-install clang-format-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES) tutorials
	$(CLANG_FORMAT) -style=file -i $(CXX_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2022 $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +

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
