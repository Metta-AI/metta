import multiprocessing
import os
import site
import sys

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as BuildExtCommand

multiprocessing.freeze_support()

try:
    site_packages = site.getsitepackages()[0]
except IndexError as err:
    site_packages = os.path.join(os.path.dirname(os.__file__), "site-packages")
    if not os.path.exists(site_packages):
        raise ImportError("site_packages is not available") from err

try:
    import numpy

    numpy_include = numpy.get_include()
except ImportError as err:
    numpy_include = os.path.join(site_packages, "numpy/core/include")
    if not os.path.exists(numpy_include):
        raise ImportError("numpy is required but not available") from err

print(f"Using NumPy include path: {numpy_include}")

# Debug flag to easily toggle development vs production settings
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "t", "yes")
print(f"Building in {'DEBUG' if DEBUG else 'RELEASE'} mode")


class CustomBuildExt(BuildExtCommand):
    def build_extensions(self):
        # Detect compiler type and add appropriate flags
        compiler_type = self.compiler.compiler_type  # ignore

        if compiler_type == "unix":
            # GCC/Clang flags
            for ext in self.extensions:
                ext.extra_compile_args.extend(["-std=c++20", "-Wno-unreachable-code"])
                if DEBUG:
                    ext.extra_compile_args.extend(["-g", "-O0"])
                else:
                    ext.extra_compile_args.extend(["-O3"])

        elif compiler_type == "msvc":
            # MSVC flags
            for ext in self.extensions:
                ext.extra_compile_args.extend(["/std:c++14", "/EHsc"])
                if DEBUG:
                    ext.extra_compile_args.extend(["/Od", "/Zi"])
                else:
                    ext.extra_compile_args.extend(["/O2"])

        # Remove -DNDEBUG from compiler flags to enable assertions when in debug mode
        if DEBUG and hasattr(self.compiler, "compiler_so"):
            try:
                if "-DNDEBUG" in self.compiler.compiler_so:
                    self.compiler.compiler_so.remove("-DNDEBUG")
            except (AttributeError, TypeError):
                pass  # Skip if compiler structure is different

        # Add include directories to all extensions
        for ext in self.extensions:
            if numpy_include not in ext.include_dirs:
                ext.include_dirs.append(numpy_include)

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
        include_dirs=[numpy_include],  # Add numpy_include here explicitly
        extra_compile_args=[],  # Will be filled by CustomBuildExt
        # Specify C++ language
        language="c++",
    )
]

# Set build directory based on debug mode
BUILD_DIR = "build_debug" if DEBUG else "build"
os.makedirs(BUILD_DIR, exist_ok=True)

# Use appropriate number of threads based on platform
if sys.platform == "linux":
    num_threads = multiprocessing.cpu_count()
elif sys.platform == "darwin":  # macOS
    num_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
else:
    num_threads = None  # Let Cython decide

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
    version="0.1",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy_include],
    package_data={
        "metta": ["*.so"],
        "metta.rl.fast_gae": ["*.so", "*.pyx", "*.pxd"],
    },
    zip_safe=False,
    cmdclass={"build_ext": CustomBuildExt},  # Use our custom build_ext class
    ext_modules=cythonize(
        ext_modules,
        build_dir=BUILD_DIR,
        nthreads=num_threads,
        annotate=DEBUG,  # Generate annotated HTML files only in debug mode
        compiler_directives=compiler_directives,
    ),
    description="Metta: AI Research Framework",
    url="https://github.com/Metta-AI/metta",
    python_requires="==3.11.7",
)
