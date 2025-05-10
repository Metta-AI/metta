import multiprocessing
import os

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

multiprocessing.freeze_support()


class Pybind11Extension(Extension):
    def __init__(self, name, sources, **kwargs):
        super().__init__(name, sources, **kwargs)
        self.include_dirs = kwargs.get("include_dirs", [])
        self.language = kwargs.get("language", "c++")
        self.extra_compile_args = kwargs.get("extra_compile_args", [])
        self.extra_link_args = kwargs.get("extra_link_args", [])


class Pybind11BuildExt(build_ext):
    def build_extensions(self):
        # Detect compiler
        compiler = self.compiler.compiler_type
        if compiler == "msvc":
            extra_compile_args = ["/std:c++20"]
            extra_link_args = []
        else:
            extra_compile_args = ["-std=c++20", "-fvisibility=hidden"]
            extra_link_args = ["-std=c++20"]

        # Add pybind11 include directory
        import pybind11

        pybind11_include = pybind11.get_include()

        for ext in self.extensions:
            ext.include_dirs.extend([numpy.get_include(), pybind11_include, "mettagrid"])
            ext.extra_compile_args.extend(extra_compile_args)
            ext.extra_link_args.extend(extra_link_args)

        build_ext.build_extensions(self)


ext_modules = [
    Pybind11Extension(
        "mettagrid.mettagrid_c",
        sources=["mettagrid/mettagrid_c.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++17"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

debug = os.getenv("DEBUG", "0") == "1"
build_dir = "build_debug" if debug else "build"
os.makedirs(build_dir, exist_ok=True)

setup(
    name="mettagrid",
    version="0.1.6",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": Pybind11BuildExt},
    install_requires=["numpy", "pybind11>=2.6.0", "gymnasium"],
)
