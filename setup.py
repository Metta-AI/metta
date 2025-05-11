import multiprocessing
import os
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
        # Remove -DNDEBUG from compiler flags to enable assertions when in debug mode
        if DEBUG and "-DNDEBUG" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-DNDEBUG")
        super().build_extensions()


ext_modules = [
    Extension(
        # Change the name to include the full path to the module
        name="metta.rl.fast_gae.fast_gae",  # This will create a fast_gae.so inside metta/rl/fast_gae
        sources=["metta/rl/fast_gae/fast_gae.pyx"],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("DEBUG", "1" if DEBUG else "0"),  # Add DEBUG macro to C++ code
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-Wno-unreachable-code"],  # Fixed typo in extra_compile_args
    )
]

# Set build directory based on debug mode
BUILD_DIR = "build_debug" if DEBUG else "build"
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
    name="metta",
    packages=find_packages(),
    cmdclass={"build_ext": CustomBuildExt},  # Use our custom build_ext class
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
