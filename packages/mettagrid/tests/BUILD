# BUILD file for mettagrid tests
load("@rules_python//python:defs.bzl", "py_test")

# Basic compiler flags
COPTS = [
    "-std=c++20",
    "-fvisibility=hidden",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    # "-fsanitize=address",  # Disabled - causes issues on macOS
]

# C++ test binaries that link with Python
cc_binary(
    name="test_stats_tracker_bin",
    srcs=["test_stats_tracker.cpp"],
    copts=COPTS + ["-Icpp/include/mettagrid", "-Itests"],
    # linkopts=["-fsanitize=address"],  # Disabled - causes issues on macOS
    linkstatic=False,
    deps=[
        "//:mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name="test_grid_object_bin",
    srcs=["test_grid_object.cpp"],
    copts=COPTS + ["-Icpp/include/mettagrid", "-Itests"],
    # linkopts=["-fsanitize=address"],  # Disabled - causes issues on macOS
    linkstatic=False,
    deps=[
        "//:mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

filegroup(
    name="mettagrid_sources",
    srcs=[
        "//cpp:bindings/mettagrid_c.cpp",
        "//cpp:bindings/mettagrid_c.hpp",
    ],
)

cc_binary(
    name="test_mettagrid_bin",
    srcs=["test_mettagrid.cpp"] + [":mettagrid_sources"],
    copts=COPTS + ["-Icpp/include/mettagrid", "-Icpp", "-Itests"],
    # linkopts=["-fsanitize=address"],  # Disabled - causes issues on macOS
    linkstatic=False,
    deps=[
        "//:mettagrid_lib",  # For pybind11 headers
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name="test_observations_bin",
    srcs=["test_observations.cpp"],
    copts=COPTS + ["-Icpp/include/mettagrid", "-Itests"],
    # linkopts=["-fsanitize=address"],  # Disabled - causes issues on macOS
    linkstatic=False,
    deps=[
        "//:mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_binary(
    name="test_clipper_bin",
    srcs=["test_clipper.cpp"],
    copts=COPTS + ["-Icpp/include/mettagrid", "-Itests"],
    # linkopts=["-fsanitize=address"],  # Disabled - causes issues on macOS
    linkstatic=False,
    deps=[
        "//:mettagrid_lib",
        "@googletest//:gtest_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

# Python test wrappers
py_test(
    name="test_stats_tracker",
    srcs=["run_test.py"],
    main="run_test.py",
    args=["--bin=test_stats_tracker_bin"],
    data=[":test_stats_tracker_bin"],
    size="small",
)

py_test(
    name="test_grid_object",
    srcs=["run_test.py"],
    main="run_test.py",
    args=["--bin=test_grid_object_bin"],
    data=[":test_grid_object_bin"],
    size="small",
)

py_test(
    name="test_mettagrid",
    srcs=["run_test.py"],
    main="run_test.py",
    args=["--bin=test_mettagrid_bin"],
    data=[":test_mettagrid_bin"],
    size="small",
)

py_test(
    name="test_observations",
    srcs=["run_test.py"],
    main="run_test.py",
    args=["--bin=test_observations_bin"],
    data=[":test_observations_bin"],
    size="small",
)

py_test(
    name="test_clipper",
    srcs=["run_test.py"],
    main="run_test.py",
    args=["--bin=test_clipper_bin"],
    data=[":test_clipper_bin"],
    size="small",
)

# Test suite that runs all tests
test_suite(
    name="tests_all",
    tests=[
        ":test_stats_tracker",
        ":test_grid_object",
        ":test_mettagrid",
        ":test_observations",
        ":test_clipper",
    ],
)
