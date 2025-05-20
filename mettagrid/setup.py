import multiprocessing
import os
<<<<<<< HEAD

import numpy
from setuptools import Extension, find_packages, setup
=======
import site

from setuptools import Extension, setup
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
from setuptools.command.build_ext import build_ext

multiprocessing.freeze_support()

<<<<<<< HEAD

class Pybind11Extension(Extension):
    def __init__(self, name, sources, **kwargs):
        super().__init__(name, sources, **kwargs)
        self.include_dirs = kwargs.get("include_dirs", [])
        self.language = kwargs.get("language", "c++")
        self.extra_compile_args = kwargs.get("extra_compile_args", [])
        self.extra_link_args = kwargs.get("extra_link_args", [])


class Pybind11BuildExt(build_ext):
    def build_extensions(self):
        # Detect compiler
        compiler = self.compiler.compiler_type
        if compiler == "msvc":
            extra_compile_args = ["/std:c++20"]
            extra_link_args = []
        else:
            extra_compile_args = ["-std=c++20", "-fvisibility=hidden"]
            extra_link_args = ["-std=c++20"]

        # Add pybind11 include directory
        import pybind11

        pybind11_include = pybind11.get_include()

        for ext in self.extensions:
            ext.include_dirs.extend([numpy.get_include(), pybind11_include, "mettagrid"])
=======
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
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
            ext.extra_compile_args.extend(extra_compile_args)
            ext.extra_link_args.extend(extra_link_args)

        build_ext.build_extensions(self)


ext_modules = [
<<<<<<< HEAD
    Pybind11Extension(
        "mettagrid.mettagrid_c",
        sources=["mettagrid/mettagrid_c.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++17"],
=======
    Extension(
        "mettagrid.mettagrid_c",
        sources=cpp_sources,
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

<<<<<<< HEAD
debug = os.getenv("DEBUG", "0") == "1"
build_dir = "build_debug" if debug else "build"
os.makedirs(build_dir, exist_ok=True)

setup(
    name="mettagrid",
    version="0.1.6",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": Pybind11BuildExt},
    install_requires=["numpy", "pybind11>=2.6.0", "gymnasium"],
=======

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
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
)
