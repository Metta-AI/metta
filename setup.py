import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ext_modules = [
    Extension(
        name="metta.rl.fast_gae",
        sources=["metta/rl/fast_gae/fast_gae.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[numpy.get_include()],
    )
]

debug = os.getenv("DEBUG", "0") == "1"
annotate = os.getenv("ANNOTATE", "0") == "1"
BUILD_DIR = "build"
if debug:
    BUILD_DIR = "build_debug"
os.makedirs(BUILD_DIR, exist_ok=True)

setup(
    name="metta",
    packages=find_packages(),
    ext_modules=cythonize(
        ext_modules,
        build_dir=BUILD_DIR,
        compiler_directives={
            "profile": True,
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
            "linetrace": debug,
            "c_string_encoding": "utf-8",
            "c_string_type": "str",
        },
        annotate=debug or annotate,
    ),
    description="",
    url="https://github.com/Metta-AI/metta",
)
