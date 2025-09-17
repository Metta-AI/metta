load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# rules_foreign_cc rule (cmake_external) will be referenced from a generated BUILD.

def _raylib_cc_repo_impl(ctx):
    version = ctx.attr.version
    sha256 = ctx.attr.sha256

    # Create a workspace-local repository that defines a cmake_external target
    # building a shared Raylib library from the downloaded sources.
    build = []
    build.append('load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")')
    build.append("")
    build.append("cmake(")
    build.append("    name = \"raylib\",")
    build.append("    lib_source = \"@raylib_src//:all\",")
    build.append("    cache_entries = {")
    build.append("        \"BUILD_SHARED_LIBS\": \"ON\",")
    build.append("        \"CMAKE_BUILD_TYPE\": \"Release\",")
    build.append("        \"RAYLIB_BUILD_EXAMPLES\": \"OFF\",")
    build.append("        \"RAYLIB_BUILD_TESTING\": \"OFF\",")
    build.append("    },")
    # The cmake macro will expose a cc_library target named 'raylib'.
    build.append(")")
    build.append("")

    ctx.file("BUILD.bazel", "\n".join(build))

raylib_cc_repo = repository_rule(
    implementation = _raylib_cc_repo_impl,
    attrs = {
        "version": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
    },
)

def _impl(mctx):
    mod = mctx.modules[0]
    t = mod.tags.cfg[0] if mod.tags.cfg else None
    version = t.version if t and t.version else "5.0"
    sha256 = t.sha256 if t and t.sha256 else ""  # fill when known

    http_archive(
        name = "raylib_src",
        urls = [
            "https://github.com/raysan5/raylib/archive/refs/tags/{}.tar.gz".format(version),
        ],
        strip_prefix = "raylib-{}".format(version),
        sha256 = sha256 if sha256 else None,
    )

    raylib_cc_repo(
        name = "raylib_built",
        version = version,
        sha256 = sha256,
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
