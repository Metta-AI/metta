import multiprocessing
import os
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

multiprocessing.freeze_support()


def build_ext(srcs, module_name=None):
    if module_name is None:
        module_name = srcs[0].replace("/", ".").replace(".pyx", "").replace(".cpp", "")
    return Extension(
        module_name,
        sources=srcs,
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-std=c++11"],
    )


ext_modules = [
    build_ext(["mettagrid/action_handler.pyx"]),
    build_ext(["mettagrid/event.pyx"]),
    build_ext(["mettagrid/grid.cpp"]),
    build_ext(["mettagrid/grid_env.pyx"]),
    build_ext(["mettagrid/grid_object.pyx"]),
    build_ext(["mettagrid/stats_tracker.cpp"]),
    build_ext(["mettagrid/observation_encoder.pyx"]),
    build_ext(["mettagrid/actions/attack.pyx"]),
    build_ext(["mettagrid/actions/attack_nearest.pyx"]),
    build_ext(["mettagrid/actions/change_color.pyx"]),
    build_ext(["mettagrid/actions/move.pyx"]),
    build_ext(["mettagrid/actions/noop.pyx"]),
    build_ext(["mettagrid/actions/rotate.pyx"]),
    build_ext(["mettagrid/actions/swap.pyx"]),
    build_ext(["mettagrid/actions/put_recipe_items.pyx"]),
    build_ext(["mettagrid/actions/get_output.pyx"]),
    build_ext(["mettagrid/objects/agent.pyx"]),
    build_ext(["mettagrid/objects/constants.pyx"]),
    build_ext(["mettagrid/objects/converter.pyx"]),
    build_ext(["mettagrid/objects/metta_object.pyx"]),
    build_ext(["mettagrid/objects/production_handler.pyx"]),
    build_ext(["mettagrid/objects/wall.pyx"]),
    build_ext(["mettagrid/mettagrid.pyx"], module_name="mettagrid.mettagrid_c"),
]

debug = os.getenv("DEBUG", "0") == "1"
annotate = os.getenv("ANNOTATE", "0") == "1"
build_dir = "build_debug" if debug else "build"
os.makedirs(build_dir, exist_ok=True)

num_threads = multiprocessing.cpu_count() if sys.platform == "linux" else None

setup(
    name="mettagrid",
    version="0.1.6",  # match pyproject.toml
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    package_data={"mettagrid": ["*.so"]},
    zip_safe=False,
    ext_modules=cythonize(
        ext_modules,
        build_dir=build_dir,
        nthreads=num_threads,
        annotate=debug or annotate,
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
    ),
)
