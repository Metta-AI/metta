# BUILD file for mettagrid C++ extension module
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_test")

# Basic compiler flags
COPTS = [
    "-std=c++20",
    "-fvisibility=hidden",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-O3",
    "-DNDEBUG",
]

# Library for tests - use pybind_library without hidden visibility
pybind_library(
    name = "mettagrid_lib",
    srcs = glob([
        "src/metta/mettagrid/*.cpp",
        "src/metta/mettagrid/**/*.cpp",
    ]),
    hdrs = glob([
        "src/metta/mettagrid/*.hpp",
        "src/metta/mettagrid/**/*.hpp",
    ]),
    copts = [
        "-std=c++20",
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        "-O3",
        "-DNDEBUG",
    ],  # Removed -fvisibility=hidden
    includes = ["src/metta", "src/metta/mettagrid"],
    visibility = ["//visibility:public"],
)

# Python extension module
pybind_extension(
    name = "mettagrid_c",
    srcs = glob([
        "src/metta/mettagrid/*.cpp",
        "src/metta/mettagrid/**/*.cpp",
        "src/metta/mettagrid/*.hpp",
        "src/metta/mettagrid/**/*.hpp",
    ]),
    copts = COPTS,
    includes = ["src/metta/mettagrid"],
    visibility = ["//visibility:public"],
)

# C++ test binaries that link with Python
cc_binary(
    name = "test_stats_tracker_bin",
    srcs = ["tests/test_stats_tracker.cpp"],
    copts = COPTS + ["-Isrc/metta", "-Isrc/metta/mettagrid", "-Itests"],
    linkstatic = False,
    deps = [
        ":mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name = "test_grid_object_bin",
    srcs = ["tests/test_grid_object.cpp"],
    copts = COPTS + ["-Isrc/metta", "-Isrc/metta/mettagrid", "-Itests"],
    linkstatic = False,
    deps = [
        ":mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name = "test_mettagrid_bin",
    srcs = ["tests/test_mettagrid.cpp"] + glob([
        "src/metta/mettagrid/*.cpp",
        "src/metta/mettagrid/**/*.cpp",
        "src/metta/mettagrid/*.hpp",
        "src/metta/mettagrid/**/*.hpp",
    ]),
    copts = COPTS + ["-Isrc/metta", "-Isrc/metta/mettagrid", "-Itests"],
    linkstatic = False,
    deps = [
        ":mettagrid_lib",  # For pybind11 headers
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name = "test_observations_bin",
    srcs = ["tests/test_observations.cpp"],
    copts = COPTS + ["-Isrc/metta", "-Isrc/metta/mettagrid", "-Itests"],
    linkstatic = False,
    deps = [
        ":mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

# Python test wrappers
py_test(
    name = "test_stats_tracker",
    srcs = ["tests/test_stats_tracker_runner.py"],
    main = "tests/test_stats_tracker_runner.py",
    data = [":test_stats_tracker_bin"],
    size = "small",
)

py_test(
    name = "test_grid_object",
    srcs = ["tests/test_grid_object_runner.py"],
    main = "tests/test_grid_object_runner.py",
    data = [":test_grid_object_bin"],
    size = "small",
)

py_test(
    name = "test_mettagrid",
    srcs = ["tests/test_mettagrid_runner.py"],
    main = "tests/test_mettagrid_runner.py",
    data = [":test_mettagrid_bin"],
    size = "small",
)

py_test(
    name = "test_observations",
    srcs = ["tests/test_observations_runner.py"],
    main = "tests/test_observations_runner.py",
    data = [":test_observations_bin"],
    size = "small",
)

# Test suite that runs all tests
test_suite(
    name = "tests_all",
    tests = [
        ":test_stats_tracker",
        ":test_grid_object",
        ":test_mettagrid",
        ":test_observations",
    ],
)

# Export the .clang-tidy config file for use by the linter
exports_files([".clang-tidy"])
