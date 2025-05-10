import multiprocessing
import os
import sys
from glob import glob  # Add this import for glob
from typing import Optional

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

multiprocessing.freeze_support()


def build_ext(srcs, module_name=None, depends=None):
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
    build_ext(["mettagrid/py_mettagrid.pyx", "mettagrid/core.cpp"], module_name="mettagrid.core", depends=all_headers)
]

debug = os.getenv("DEBUG", "0") == "1"
annotate = os.getenv("ANNOTATE", "0") == "1"
build_dir = "build_debug" if debug else "build"
os.makedirs(build_dir, exist_ok=True)
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
        build_dir=build_dir,
        nthreads=num_threads,  # type: ignore[reportArgumentType] -- Pylance is wrong. We want "None" when not on linux.
        annotate=debug or annotate,
        compiler_directives={
            "language_level": "3",
            "embedsignature": debug,
            "annotation_typing": debug,
            "cdivision": debug,
            "boundscheck": debug,
            "wraparound": debug,
            "initializedcheck": debug,
            "nonecheck": debug,
            "overflowcheck": debug,
            "overflowcheck.fold": debug,
            "profile": debug,
            "linetrace": debug,
            "c_string_encoding": "utf8",
            "c_string_type": "str",
        },
    ),
)
