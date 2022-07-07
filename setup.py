import os
import pathlib
import sys
import shutil

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).absolute().parent

sys.path.insert(0, str(HERE / 'torchopt'))
import version  # noqa


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def copy(self, extdir):
        from distutils.file_util import copy_file

        for op_path in pathlib.Path(extdir).iterdir():
            if not op_path.is_dir():
                continue
            for file in op_path.iterdir():
                if str(file).rpartition('.')[-1] == 'so':
                    copy_file(file, HERE / 'torchopt' / '_lib')

    def build_extensions(self):
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
                f'-DPYTHON_INCLUDE_DIR={PYTHON_INCLUDE_DIR}',
                f'-DTORCH_INCLUDE_PATH={TORCH_INCLUDE_PATH}',
                f'-DTORCH_LIBRARY_PATH={TORCH_LIBRARY_PATH}',
            ]

            build_args = ['--config', config, '--', '-j4']

            try:
                os.chdir(build_temp)
                self.spawn(['cmake', ext.cmake_lists_dir] + cmake_args)
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
    url='https://github.com/metaopt/torchopt',
    packages=find_packages(include=['torchopt', 'torchopt.*']),
    package_data={'sharedlib': ['_lib/*.so']},
    include_package_data=True,
    cmdclass={'build_ext': cmake_build_ext},
    ext_modules=[
        CMakeExtension('torchopt._lib.adam_op', cmake_lists_dir=HERE)
    ],
    setup_requires=[  # for `torch.utils.cpp_extension`
        'torch==1.11',
        'numpy',
    ],
    install_requires=[
        'torch==1.11',
        'jax[cpu]',
        'numpy',
        'graphviz',
    ],
    python_requires='>=3.7'
)
