import os
import pathlib
import shutil
import sys

from setuptools import find_packages, setup


try:
    from pybind11.setup_helpers import Pybind11Extension as Extension
    from pybind11.setup_helpers import build_ext
except ImportError:
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).absolute().parent

sys.path.insert(0, str(HERE / 'torchopt'))
import version  # noqa


class CMakeExtension(Extension):
    def __init__(self, name, source_dir='.', **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.source_dir = os.path.abspath(source_dir)


class cmake_build_ext(build_ext):
    def copy(self, extdir):
        for op_path in pathlib.Path(extdir).iterdir():
            if not op_path.is_dir():
                continue
            for file in op_path.iterdir():
                if str(file).rpartition('.')[-1] == 'so':
                    shutil.copy(file, HERE / 'torchopt' / '_lib')

    def build_extensions(self):
        import pybind11
        from torch.utils import cpp_extension

        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            print(self.get_ext_fullpath(ext.name))

            PYTHON_INCLUDE_DIR = ';'.join(self.include_dirs)
            TORCH_INCLUDE_PATH = ';'.join(cpp_extension.include_paths())
            TORCH_LIBRARY_PATH = ';'.join(cpp_extension.library_paths())

            cmake_args = [
                f'-DCMAKE_BUILD_TYPE={config}',
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={extdir}',
                f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{config.upper()}={self.build_temp}',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
                f'-DPYBIND11_CMAKE_DIR={pybind11.get_cmake_dir()}',
                f'-DPYTHON_INCLUDE_DIR={PYTHON_INCLUDE_DIR}',
                f'-DTORCH_INCLUDE_PATH={TORCH_INCLUDE_PATH}',
                f'-DTORCH_LIBRARY_PATH={TORCH_LIBRARY_PATH}',
            ]

            build_args = ['--config', config]

            if (
                'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ
                and hasattr(self, 'parallel') and self.parallel
            ):
                build_args.append(f'-j{self.parallel}')

            try:
                os.chdir(build_temp)
                self.spawn(['cmake', ext.source_dir] + cmake_args)
                if not self.dry_run:
                    self.spawn(['cmake', '--build', '.'] + build_args)
                self.copy(extdir)
            finally:
                os.chdir(HERE)


setup(
    name='torchopt',
    version=version.__version__,
    author='TorchOpt Contributors',
    author_email='jieren9806@gmail.com, xidong.feng.20@ucl.ac.uk, benjaminliu.eecs@gmail.com',
    description='A Jax-style optimizer for PyTorch.',
    license='Apache License Version 2.0',
    keywords='Meta-Learning, PyTorch, Optimizer',
    url='https://github.com/metaopt/TorchOpt',
    packages=find_packages(include=['torchopt', 'torchopt.*']),
    package_data={'sharedlib': ['_lib/*.so']},
    include_package_data=True,
    cmdclass={'build_ext': cmake_build_ext},
    ext_modules=[
        CMakeExtension('torchopt._lib.adam_op', source_dir=HERE)
    ],
    setup_requires=[  # for `torch.utils.cpp_extension`
        'torch == 1.12',
        'numpy',
        'pybind11',
    ],
    install_requires=[
        'torch == 1.12',
        'jax[cpu] >= 0.3',
        'numpy',
        'graphviz',
        'typing-extensions',
    ],
    python_requires='>= 3.7'
)
