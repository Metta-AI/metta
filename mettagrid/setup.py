import multiprocessing
import os
import sys
from glob import glob  # Add this import for glob
from typing import Optional

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as BuildExtCommand  # Import the build_ext class

multiprocessing.freeze_support()


class CustomBuildExt(BuildExtCommand):
    def build_extensions(self):
        # Remove -DNDEBUG from compiler flags to enable assertions
        if "-DNDEBUG" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-DNDEBUG")
        super().build_extensions()


def create_extension(srcs, module_name=None, depends=None):
    if module_name is None:
        module_name = srcs[0].replace("/", ".").replace(".pyx", "").replace(".cpp", "")
    # Add the project root directory to include paths so C++ files can find each other
    return Extension(
        module_name,
        sources=srcs,
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-std=c++23", "-Wno-unreachable-code"],
        include_dirs=[numpy.get_include(), ".", "./mettagrid", "./third_party"],
        depends=depends or [],
    )


# Get all header files
mettagrid_headers = glob("mettagrid/*.hpp")
action_headers = glob("mettagrid/actions/*.hpp")
object_headers = glob("mettagrid/objects/*.hpp")
all_headers = mettagrid_headers + action_headers + object_headers

ext_modules = [
    create_extension(
        ["mettagrid/py_mettagrid.pyx", "mettagrid/core.cpp"], module_name="mettagrid.core", depends=all_headers
    )
]


os.makedirs("build", exist_ok=True)
num_threads: Optional[int] = multiprocessing.cpu_count() if sys.platform == "linux" else None

setup(
    name="mettagrid",
    version="0.2",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    ext_modules=cythonize(
        ext_modules,
        build_dir="build",
        nthreads=num_threads,  # type: ignore[reportArgumentType] -- Pylance is wrong. We want "None" when not on linux.
        annotate=False,  # Generate annotated HTML files to see Python → C translation
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,  # Include docstrings and signatures in compiled module
            "annotation_typing": True,  # Enable type annotations
            # These are the critical ones for catching errors:
            "boundscheck": True,  # Check array bounds (catches index errors)
            "wraparound": True,  # Handle negative indices correctly
            "initializedcheck": True,  # Check if memoryviews are initialized
            "nonecheck": True,  # Check if arguments are None
            "overflowcheck": True,  # Check for C integer overflow
            "overflowcheck.fold": True,  # Also check operations folded by the compiler
            "cdivision": False,  # Check for division by zero (slows code down)
            # For performance debugging:
            "profile": True,  # Enable profiling
            "linetrace": True,  # Enable line tracing for coverage tools
            "c_string_encoding": "utf8",
            "c_string_type": "str",
        },
    ),
)
