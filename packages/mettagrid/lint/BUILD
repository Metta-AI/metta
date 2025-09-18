# BUILD file for custom lint rules

load(":clang_tidy.bzl", "clang_tidy_test", "clang_tidy_all")

# Wrapper for system clang-tidy
sh_binary(
    name = "clang_tidy_wrapper",
    srcs = ["clang_tidy_wrapper.sh"],
    visibility = ["//visibility:public"],
)

# Run clang-tidy on all C++ targets and fail if there are issues
clang_tidy_test(
    name = "clang_tidy",
    deps = [
        "//:mettagrid_lib",
        "//:mettagrid_c",
    ],
)

# Run clang-tidy and generate a report (doesn't fail)
clang_tidy_all(
    name = "clang_tidy_report",
    deps = [
        "//:mettagrid_lib",
        "//:mettagrid_c",
    ],
)

# Alias for convenience
alias(
    name = "lint",
    actual = ":clang_tidy",
    visibility = ["//visibility:public"],
)
