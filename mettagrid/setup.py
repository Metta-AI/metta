import multiprocessing
import os
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

multiprocessing.freeze_support()


def find_source_files(directory):
    source_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                source_files.append(os.path.join(root, file))
    return source_files


class CustomBuildExt(build_ext):
    def run(self):
        # First, run the normal extension build
        super().run()

        # Then create individual object files for testing
        self.build_individual_objects()

    def build_individual_objects(self):
        """Build individual .o files that tests can link against"""
        print("Building individual object files for testing...")

        # Create build directories
        build_dir = Path("build/mettagrid")
        build_dir.mkdir(parents=True, exist_ok=True)

        # Source files and directories
        cpp_sources = find_source_files("mettagrid")
        project_hdr = os.path.abspath("mettagrid")

        # Compiler settings (match the extension settings)
        compile_args = [
            "g++",
            "-std=c++20",
            "-Wall",
            "-g",
            "-fPIC",  # Important for shared libraries
            "-fvisibility=hidden",
            f"-I{project_hdr}",
            "-c",  # Compile only, don't link
        ]

        # Add Python and numpy includes
        import numpy
        import pybind11

        python_includes = pybind11.get_cmake_dir().replace("pybind11/share/cmake/pybind11", "")
        compile_args.extend(
            [
                f"-I{pybind11.get_include()}",
                f"-I{numpy.get_include()}",
            ]
        )

        # Build each source file to an object file
        for cpp_file in cpp_sources:
            source_path = Path(cpp_file)
            object_name = source_path.stem + ".o"
            object_path = build_dir / object_name

            cmd = compile_args + [str(source_path), "-o", str(object_path)]

            print(f"Building {object_path}...")
            try:
                subprocess.run(cmd, check=True, cwd=".")
                print(f"✅ Built {object_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to build {object_path}: {e}")
                raise


# Get all CPP files (only .cpp files are used for sources)
cpp_sources = find_source_files("mettagrid")
print(f"Found {len(cpp_sources)} source files: {cpp_sources}")

project_hdr = os.path.abspath("mettagrid")

ext_modules = [
    Pybind11Extension(
        "mettagrid.mettagrid_c",
        cpp_sources,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[project_hdr],
        extra_compile_args=[
            "-std=c++20",
            "-fvisibility=hidden",
        ],
        extra_link_args=[
            "-std=c++20",
        ],
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
    cmdclass={"build_ext": CustomBuildExt},  # Use our custom build class
    python_requires="==3.11.7",
)
