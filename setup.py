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
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-std=c++11'] # Add C++11 flag to fix defaulted function definition error
    )

ext_modules = [

    build_ext(["mettagrid/action.pyx"]),
    build_ext(["mettagrid/event.pyx"]),
    build_ext(["mettagrid/grid.cpp"]),
    build_ext(["mettagrid/grid_env.pyx"]),
    build_ext(["mettagrid/grid_object.pyx"]),
    build_ext(["mettagrid/stats_tracker.cpp"]),

    build_ext(["mettagrid/observation_encoder.pyx"]),
    build_ext(["mettagrid/actions/actions.pyx"]),
    build_ext(["mettagrid/actions/attack.pyx"]),
    build_ext(["mettagrid/actions/attack_nearest.pyx"]),
    build_ext(["mettagrid/actions/change_color.pyx"]),
    build_ext(["mettagrid/actions/move.pyx"]),
    build_ext(["mettagrid/actions/noop.pyx"]),
    build_ext(["mettagrid/actions/rotate.pyx"]),
    build_ext(["mettagrid/actions/swap.pyx"]),
    build_ext(["mettagrid/actions/put_recipe_items.pyx"]),
    build_ext(["mettagrid/actions/get_output.pyx"]),

    build_ext(["mettagrid/objects/altar.pyx"]),
    build_ext(["mettagrid/objects/agent.pyx"]),
    build_ext(["mettagrid/objects/armory.pyx"]),
    build_ext(["mettagrid/objects/constants.pyx"]),
    build_ext(["mettagrid/objects/converter.pyx"]),
    build_ext(["mettagrid/objects/factory.pyx"]),
    build_ext(["mettagrid/objects/generator.pyx"]),
    build_ext(["mettagrid/objects/lab.pyx"]),
    build_ext(["mettagrid/objects/lasery.pyx"]),
    build_ext(["mettagrid/objects/metta_object.pyx"]),
    build_ext(["mettagrid/objects/mine.pyx"]),
    build_ext(["mettagrid/objects/production_handler.pyx"]),
    build_ext(["mettagrid/objects/wall.pyx"]),
    build_ext(["mettagrid/mettagrid.pyx"], "mettagrid.mettagrid_c"),
]

debug = os.getenv('DEBUG', '0') == '1'
annotate = os.getenv('ANNOTATE', '0') == '1'

build_dir = 'build'
if debug:
    build_dir = 'build_debug'

os.makedirs(build_dir, exist_ok=True)

num_threads = multiprocessing.cpu_count() if sys.platform == 'linux' else None

setup(
    name='metta',
    version='0.1',
    packages=find_packages(),
    nthreads=num_threads,
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
