workspace(name="metta")


# Python and pybind11 dependencies
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Python rules
http_archive(
    name="rules_python",
    sha256="a30abdfc7126d497a7698c29c46ea9901c6392d6ed315171a6df5ce433aa4502",
    strip_prefix="rules_python-0.16.1",
    url="https://github.com/bazelbuild/rules_python/archive/refs/tags/0.16.1.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_install")

# Configure Python interpreter
pip_install(
    name="python_deps",
    requirements="//third_party:requirements.txt",
)

http_archive(
    name="com_google_googletest",
    sha256="81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",  # googletest-release-1.12.1
    strip_prefix="googletest-release-1.12.1",
    urls=["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"],
)
