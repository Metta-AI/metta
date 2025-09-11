# Benchmarks for mettagrid

cc_binary(
    name = "test_mettagrid_env_benchmark",
    srcs = ["test_mettagrid_env_benchmark.cpp"],
    copts = [
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Isrc/metta",
        "-Isrc/metta/mettagrid",
    ],
    linkstatic = True,
    deps = [
        "//:mettagrid_lib",
        "@google_benchmark//:benchmark_main",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)