# Benchmarks:
# bazel build //mettagrid:grid_object_benchmark

# Tests:
# bazel test //mettagrid:grid_object_test

package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library_test")

pybind_library(
    name = "mettagrid_lib",
    srcs = glob([
        "mettagrid/**/*.cpp",
    ]),
    hdrs = glob([
        "mettagrid/**/*.hpp",
    ]),
    includes = ["mettagrid"],
    deps = [
        "//mettagrid/third_party:nlohmann_json",
    ],
)

cc_binary(
    name = "grid_object_benchmark",
    srcs = ["benchmarks/grid_object_benchmark.cpp"],
    deps = [
        ":mettagrid_lib",
        "@google_benchmark//:benchmark",
    ],
)

pybind_library_test(
    name = "all_cpp_tests",
    srcs = glob(["tests/*.cpp"]),
    deps = [
        ":mettagrid_lib",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
