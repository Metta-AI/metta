workspace(name = "metta")


# Python and pybind11 dependencies
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Python rules
http_archive(
    name = "rules_python",
    sha256 = "a30abdfc7126d497a7698c29c46ea9901c6392d6ed315171a6df5ce433aa4502",
    strip_prefix = "rules_python-0.16.1",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.16.1.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_install")

# Configure Python interpreter
pip_install(
    name = "python_deps",
    requirements = "//third_party:requirements.txt",
)

# pybind11 rules
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-master",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/master.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.10.1",
    urls = ["https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
