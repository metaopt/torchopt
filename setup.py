import contextlib
import os
import pathlib
import platform
import re
import shutil
import sys
import sysconfig
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = pathlib.Path(__file__).absolute().parent


class CMakeExtension(Extension):
    def __init__(self, name, source_dir='.', target=None, **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.source_dir = os.path.abspath(source_dir)
        self.target = target if target is not None else name.rpartition('.')[-1]


class cmake_build_ext(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        from torch.utils import cpp_extension

        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

        ext_path = pathlib.Path(self.get_ext_fullpath(ext.name)).absolute()
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={ext_path.parent}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{config.upper()}={build_temp}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPYTHON_INCLUDE_DIR={sysconfig.get_path("platinclude")}',
            f'-DTORCH_INCLUDE_PATH={";".join(cpp_extension.include_paths())}',
            f'-DTORCH_LIBRARY_PATH={";".join(cpp_extension.library_paths())}',
        ]

        if platform.system() == 'Darwin':
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r'-arch (\S+)', os.environ.get('ARCHFLAGS', ''))
            if archs:
                cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}')

        try:
            import pybind11

            cmake_args.append(f'-DPYBIND11_CMAKE_DIR={pybind11.get_cmake_dir()}')
        except ImportError:
            pass

        build_args = ['--config', config]
        if (
            'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ
            and hasattr(self, 'parallel')
            and self.parallel
        ):
            build_args.extend(['--parallel', str(self.parallel)])
        else:
            build_args.append('--parallel')

        build_args.extend(['--target', ext.target, '--'])

        cwd = os.getcwd()
        try:
            os.chdir(build_temp)
            self.spawn([cmake, ext.source_dir, *cmake_args])
            if not self.dry_run:
                self.spawn([cmake, '--build', '.', *build_args])
        finally:
            os.chdir(cwd)


@contextlib.contextmanager
def vcs_version(name, path):
    path = pathlib.Path(path).absolute()
    assert path.is_file()
    module_spec = spec_from_file_location(name=name, location=path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = sys.modules.get(name)
    if module is None:
        module = module_from_spec(module_spec)
        sys.modules[name] = module
    module_spec.loader.exec_module(module)

    if module.__release__:
        yield module
        return

    content = None
    try:
        try:
            content = path.read_text(encoding='utf-8')
            path.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f'__version__ = {module.__version__!r}',
                    string=content,
                ),
                encoding='utf-8',
            )
        except OSError:
            content = None

        yield module
    finally:
        if content is not None:
            with path.open(mode='wt', encoding='utf-8', newline='') as file:
                file.write(content)


CIBUILDWHEEL = os.getenv('CIBUILDWHEEL', '0') == '1'
LINUX = platform.system() == 'Linux'
MACOS = platform.system() == 'Darwin'
WINDOWS = platform.system() == 'Windows'
ext_kwargs = {
    'cmdclass': {'build_ext': cmake_build_ext},
    'ext_modules': [
        CMakeExtension(
            'torchopt._C',
            source_dir=HERE,
            optional=not (LINUX and CIBUILDWHEEL),
        ),
    ],
}

TORCHOPT_NO_EXTENSIONS = bool(os.getenv('TORCHOPT_NO_EXTENSIONS', '')) or WINDOWS or MACOS
if TORCHOPT_NO_EXTENSIONS:
    ext_kwargs.clear()


with vcs_version(name='torchopt.version', path=(HERE / 'torchopt' / 'version.py')) as version:
    setup(
        name='torchopt',
        version=version.__version__,
        **ext_kwargs,
    )
