import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Define the extension module
ext_modules = [
    Extension(
        "cython_processor",
        sources=["cython_processor.pyx", "cpp_processor.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
]

# Setup
setup(
    name="cython_processor",
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy"],
)
