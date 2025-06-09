import multiprocessing
import os
import sys
from glob import glob
from typing import Optional

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as BuildExtCommand

multiprocessing.freeze_support()

# Debug flag to easily toggle development vs production settings
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "t", "yes")


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
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("DEBUG", "1" if DEBUG else "0"),  # Add DEBUG macro to C++ code
        ],
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
        ["mettagrid/py_mettagrid.pyx", "mettagrid/core.cpp"], module_name="mettagrid.mettagrid_c", depends=all_headers
    )
]

BUILD_DIR = "build"
os.makedirs(BUILD_DIR, exist_ok=True)
num_threads: Optional[int] = multiprocessing.cpu_count() if sys.platform == "linux" else None

# Configure compiler directives based on DEBUG setting
compiler_directives = {
    "language_level": "3",
    "embedsignature": True,  # Include docstrings and signatures in compiled module
    "annotation_typing": True,  # Enable type annotations
    "c_string_encoding": "utf8",
    "c_string_type": "str",
}

# Add debug-specific directives when DEBUG is True
if DEBUG:
    compiler_directives.update(
        {
            # Error catching settings
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
        }
    )
else:
    # Production settings for better performance
    compiler_directives.update(
        {
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "overflowcheck": False,
            "cdivision": True,
            "profile": False,
            "linetrace": False,
        }
    )

setup(
    name="mettagrid",
    version="0.2",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=cythonize(
        ext_modules,
        build_dir=BUILD_DIR,
        nthreads=num_threads,  # type: ignore[reportArgumentType] -- Pylance is wrong.
        annotate=False,  # Generate annotated HTML files to see Python → C translation
        compiler_directives=compiler_directives,
    ),
    description="",
    url="https://github.com/Metta-AI/metta",
)
