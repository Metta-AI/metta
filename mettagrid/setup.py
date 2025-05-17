import multiprocessing
import os
import site

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

multiprocessing.freeze_support()

site_packages = site.getsitepackages()[0]

# Get NumPy and pybind11 includes
numpy_include = os.path.join(site_packages, "numpy/core/include")
pybind11_include = os.path.join(site_packages, "pybind11/include")


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


class CustomBuildExt(build_ext):
    def build_extensions(self):
        extra_compile_args = ["-std=c++20", "-fvisibility=hidden"]
        extra_link_args = ["-std=c++20"]

        # Add includes
        for ext in self.extensions:
            ext.include_dirs.extend([numpy_include, pybind11_include, "mettagrid"])
            ext.extra_compile_args.extend(extra_compile_args)
            ext.extra_link_args.extend(extra_link_args)

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "mettagrid.mettagrid_c",
        sources=cpp_sources,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
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
)
