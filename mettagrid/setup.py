import multiprocessing
import os

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
    cmdclass={"build_ext": build_ext},
    python_requires="==3.11.7",
)
