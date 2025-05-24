import multiprocessing
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

multiprocessing.freeze_support()

# Debug flag to easily toggle development vs production settings
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "t", "yes")


class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Remove -DNDEBUG from compiler flags to enable assertions when DEBUG=True
        if DEBUG:
            for ext in self.extensions:
                # Remove NDEBUG from define_macros
                if hasattr(ext, "define_macros"):
                    ext.define_macros = [macro for macro in ext.define_macros if macro[0] != "NDEBUG"]

                # Remove from extra compile args
                if hasattr(ext, "extra_compile_args"):
                    ext.extra_compile_args = [arg for arg in ext.extra_compile_args if not arg.startswith("-DNDEBUG")]

            # Remove from compiler flags if present
            if hasattr(self.compiler, "compiler_so"):
                self.compiler.compiler_so = [arg for arg in self.compiler.compiler_so if not arg.startswith("-DNDEBUG")]

            print("DEBUG MODE: Assertions enabled, optimizations reduced")
        else:
            print("RELEASE MODE: Optimizations enabled, assertions disabled")

        super().build_extensions()


def find_source_files(directory):
    source_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                source_files.append(os.path.join(root, file))
    return source_files


# Get all CPP files
cpp_sources = find_source_files("mettagrid")
print(f"Found {len(cpp_sources)} source files: {cpp_sources}")

project_hdr = os.path.abspath("mettagrid")

# Configure compilation flags based on DEBUG setting
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
extra_compile_args = ["-std=c++20", "-fvisibility=hidden"]
extra_link_args = ["-std=c++20"]

if DEBUG:
    # Debug mode: Enable assertions, add debug info, reduce optimization
    define_macros.extend(
        [
            ("DEBUG", "1"),
            # Explicitly undefine NDEBUG to ensure assertions work
            ("UNDEBUG", "1"),
        ]
    )
    extra_compile_args.extend(
        [
            "-g",  # Add debug symbols
            "-O0",  # No optimization for easier debugging
            "-UNDEBUG",  # Undefine NDEBUG (redundant but explicit)
            "-Wall",  # Enable warnings
            "-Wextra",  # Extra warnings
            "-fno-omit-frame-pointer",  # Keep frame pointers for better stack traces
        ]
    )
else:
    # Release mode: Optimize for performance
    define_macros.append(("NDEBUG", "1"))
    extra_compile_args.extend(
        [
            "-O3",  # Maximum optimization
            "-DNDEBUG",  # Explicitly define NDEBUG to disable assertions
        ]
    )

print(f"DEBUG mode: {DEBUG}")
print(f"Compile flags: {extra_compile_args}")
print(f"Define macros: {define_macros}")

ext_modules = [
    Pybind11Extension(
        "mettagrid.mettagrid_c",
        cpp_sources,
        define_macros=define_macros,
        include_dirs=[project_hdr],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# Setup
setup(
    name="mettagrid",
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    version="0.1.6",
    packages=["mettagrid"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    python_requires="==3.11.7",
)
