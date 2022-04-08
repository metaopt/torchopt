import os
import sys
import pathlib
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension


class MyBuild(build_ext):
    def run(self):
        self.build_cmake()

    def copy(self, build_temp):
        from distutils.file_util import copy_file
        cwd = str(pathlib.Path().absolute())
        src = os.path.join('.', build_temp, 'src')
        ops = os.listdir(src)
        for op in ops:
            op_path = os.path.join(src, op)
            if not os.path.isdir(op_path):
                continue
            files = os.listdir(op_path)
            for file in files:
                if file.split('.')[-1] == 'so':
                    copy_file(os.path.join(op_path, file), os.path.join(
                        cwd, 'TorchOpt', '_lib'))

    def build_cmake(self):
        cwd = pathlib.Path().absolute()

        build_temp = f"{pathlib.Path(self.build_temp)}"
        os.makedirs(build_temp, exist_ok=True)

        config = "Debug" if self.debug else "Release"

        PYTHON_INCLUDE_DIR = ""
        for path in self.include_dirs:
            PYTHON_INCLUDE_DIR += path + ';'

        TORCH_INCLUDE_PATH = ""
        for path in cpp_extension.include_paths():
            TORCH_INCLUDE_PATH += path + ';'

        TORCH_LIBRARY_PATH = ""
        for path in cpp_extension.library_paths():
            TORCH_LIBRARY_PATH += path + ';'

        cmake_args = [
            "-DPYTHON_INCLUDE_DIR=" + PYTHON_INCLUDE_DIR,
            "-DTORCH_INCLUDE_PATH=" + TORCH_INCLUDE_PATH,
            "-DTORCH_LIBRARY_PATH=" + TORCH_LIBRARY_PATH,
            "-DCMAKE_BUILD_TYPE=" + config
        ]

        build_args = [
            "--config", config,
            "--", "-j4"
        ]

        os.chdir(build_temp)
        self.spawn(["cmake", f"{str(cwd)}"] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))
        self.copy(build_temp)


class download_shared():
    def __init__(self):
        import urllib
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f"setup.py at {dir_path}")
        print("downloading shared libraries")
        op_urls = []
        if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
            op_urls.append(
                "https://torchopt.oss-cn-beijing.aliyuncs.com/torch1_11/adam_op.cpython-38-x86_64-linux-gnu.so")
        elif sys.version_info >= (3, 9) and sys.version_info < (3, 10):
            op_urls.append(
                "https://torchopt.oss-cn-beijing.aliyuncs.com/torch1_11/adam_op.cpython-39-x86_64-linux-gnu.so")

        if len(op_urls) == 0:
            import warnings
            warnings.warn("no pre-compiled libraries for you python version")
            return

        for url in op_urls:
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(dir_path, 'TorchOpt', '_lib', filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
        print("shared libraries downloaded")


if 'build_from_source' not in sys.argv:
    download_shared()

setup(
    name="TorchOpt",
    version="0.4.0",
    author="Jie Ren",
    author_email="jieren9806@gmail.com",
    description="A Jax-style optimizer.",
    license="Apache License Version 2.0",
    keywords="meta learning",
    url="https://github.com/metaopt/TorchOpt",
    packages=find_packages(),
    package_data={"": ["_lib/*.so"]},
    include_package_data=True,
    cmdclass={'build_from_source': MyBuild},
    install_requires=[
        'jax[cpu]',
        'torch==1.11',
        'graphviz',
    ],
)
