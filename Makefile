print-%  : ; @echo $* = $($*)
SHELL          = /bin/bash
PROJECT_NAME   = TorchOpt
PROJECT_PATH   = ${PROJECT_NAME}/
PROJECT_FOLDER = $(PROJECT_NAME) examples include src tests
PYTHON_FILES   = $(shell find . -type f -name "*.py")
CPP_FILES      = $(shell find . -type f -name "*.h" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.cu")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
PATH           := $(HOME)/go/bin:$(PATH)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)


flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, isort)
	$(call check_install, yapf)

mypy-install:
	$(call check_install, mypy)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format-11 || sudo apt-get install -y clang-format-11

clang-tidy-install:
	command -v clang-tidy || sudo apt-get install -y clang-tidy

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install, doc8)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

pytest-install:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	$(call check_install, pytest_xdist)


# test

pytest: pytest-install
	pytest tests --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing --color=yes

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

py-format: py-format-install
	isort --check $(PYTHON_FILES) && yapf -ir $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	clang-format-11 --style=file -i $(CPP_FILES) -n --Werror

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2022 -check $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

lint: flake8 py-format clang-format cpplint mypy

format: py-format-install clang-format-install
	isort $(PYTHON_FILES)
	yapf -ir $(PYTHON_FILES)
	clang-format-11 -style=file -i $(CPP_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2022 $(PROJECT_FOLDER)

