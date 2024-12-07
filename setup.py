print("Starting setup.py")

from sympy import li
from setuptools import Extension, setup, find_packages, Command
from Cython.Build import cythonize
import numpy
import os
import multiprocessing
import sys
multiprocessing.freeze_support()

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
    build_ext(["mettagrid/observation_encoder.pyx"]),
    build_ext(["mettagrid/actions/actions.pyx"]),
    build_ext(["mettagrid/actions/attack.pyx"]),
    build_ext(["mettagrid/actions/gift.pyx"]),
    build_ext(["mettagrid/actions/move.pyx"]),
    build_ext(["mettagrid/actions/noop.pyx"]),
    build_ext(["mettagrid/actions/rotate.pyx"]),
    build_ext(["mettagrid/actions/shield.pyx"]),
    build_ext(["mettagrid/actions/swap.pyx"]),
    build_ext(["mettagrid/actions/use.pyx"]),
    build_ext(["mettagrid/mettagrid.pyx"], "mettagrid.mettagrid_c"),
]

debug = os.getenv('DEBUG', '0') == '1'
annotate = os.getenv('ANNOTATE', '0') == '1'

build_dir = 'build'
if debug:
    build_dir = 'build_debug'

os.makedirs(build_dir, exist_ok=True)

num_threads = multiprocessing.cpu_count() if sys.platform == 'linux' else None

print("Running setup.py")

setup(
    name='metta',
    version='0.1',
    packages=find_packages(),
    nthreads=num_threads,
    install_requires=[
        "hydra-core>=1.3.2",
        "jmespath>=1.0.1",
        "matplotlib>=3.9.2",
        "pettingzoo>=1.24.1",
        "pynvml>=11.5.3",
        "pytest>=8.3.3",
        "PyYAML>=6.0.2",
        "raylib>=5.5.0.1",
        "rich>=13.9.4",
        "scipy>=1.14.1",
        "tabulate>=0.9.0",
        "tensordict>=0.6.2",
        "torchrl>=0.6.0",
        "termcolor>=2.4.0",
        "wandb>=0.18.3",
        "wandb-core>=0.17.0b11",
        "pandas>=2.2.3",
        "tqdm>=4.67.1",
        # Sibling packages:
        "pufferlib",
        "puffergrid",
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
        build_dir=build_dir,
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


print("Done running setup.py")
