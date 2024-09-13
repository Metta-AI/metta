from sympy import li
from setuptools import Extension, setup, find_packages, Command
import subprocess
from Cython.Build import cythonize
import numpy
import os


def build_ext(srcs, module_name=None):
    if module_name is None:
        module_name = srcs[0].replace("/", ".").replace(".pyx", "").replace(".cpp", "")
    return Extension(
        module_name,
        srcs,
        # include_dirs=["third_party/puffergrid"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )

ext_modules = [
    build_ext(["mettagrid/objects.pyx"]),
    build_ext(["mettagrid/actions/actions.pyx"]),
    build_ext(["mettagrid/actions/attack.pyx"]),
    build_ext(["mettagrid/actions/gift.pyx"]),
    build_ext(["mettagrid/actions/move.pyx"]),
    build_ext(["mettagrid/actions/noop.pyx"]),
    build_ext(["mettagrid/actions/rotate.pyx"]),
    build_ext(["mettagrid/actions/shield.pyx"]),
    build_ext(["mettagrid/actions/use.pyx"]),
    build_ext(["mettagrid/mettagrid.pyx"], "mettagrid.mettagrid_c"),
]

debug = os.getenv('DEBUG', '0') == '1'
annotate = os.getenv('ANNOTATE', '0') == '1'

build_dir = 'build'
if debug:
    build_dir = 'build_debug'

os.makedirs(build_dir, exist_ok=True)

setup(
    name='metta',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "jmespath",
        "matplotlib",
        # "numpy==2.0.0",
        "pettingzoo",
	    "pufferlib",
        "puffergrid",
        "pynvml",
        "pytest",
        "PyYAML",
        "raylib",
        "rich",
        "scipy",
        "tabulate",
        "tensordict",
        "torchrl",
    ],
    entry_points={
        'console_scripts': [
            # If you want to create any executable scripts in your package
            # For example: 'script_name = module:function'
        ]
    },
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(
        ext_modules,
        build_dir='build',
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
            "c_string_encoding": "utf-8",
            "c_string_type": "str",
        },
        annotate=debug or annotate,
    ),
)
