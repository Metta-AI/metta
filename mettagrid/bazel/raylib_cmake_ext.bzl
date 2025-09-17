load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _impl(mctx):
    mod = mctx.modules[0]
    t = mod.tags.cfg[0] if mod.tags.cfg else None
    version = t.version if t and t.version else "5.5"
    sha256 = t.sha256 if t and t.sha256 else None

    build_content = "\n".join([
        'load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")',
        "",
        "filegroup(",
        "    name = \"all_srcs\",",
        "    srcs = glob([\"**\"]),",
        "    visibility = [\"//visibility:public\"],",
        ")",
        "",
        "cmake(",
        "    name = \"raylib\",",
        "    lib_source = \":all_srcs\",",
        "    cache_entries = {",
        "        \"CMAKE_BUILD_TYPE\": \"Release\",",
        "        \"BUILD_SHARED_LIBS\": \"ON\",",
        "        \"RAYLIB_BUILD_EXAMPLES\": \"OFF\",",
        "        \"RAYLIB_BUILD_TESTING\": \"OFF\",",
        "    },",
        "    install = True,",
        "    out_shared_libs = [\"libraylib.dylib\"],",
        "    out_lib_dir = \"lib\",",
        "    out_include_dir = \"include\",",
        "    visibility = [\"//visibility:public\"],",
        ")",
        "",
    ])

    http_archive(
        name = "raylib_src",
        urls = [
            "https://github.com/raysan5/raylib/archive/refs/tags/{}.tar.gz".format(version),
        ],
        strip_prefix = "raylib-{}".format(version),
        sha256 = sha256,
        build_file_content = build_content,
    )

raylib_cmake_ext = module_extension(
    implementation = _impl,
    tag_classes = {
        "cfg": tag_class(attrs = {
            "version": attr.string(),
            "sha256": attr.string(),
        }),
    },
)
