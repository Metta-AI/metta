def _raylib_repo_impl(ctx):
    # Determine Homebrew prefix to use
    hb1 = ctx.attr.opt_homebrew
    hb2 = ctx.attr.usr_local
    prefix = None
    if hb1 and ctx.path(hb1 + "/include").exists:
        prefix = hb1
    elif hb2 and ctx.path(hb2 + "/include").exists:
        prefix = hb2

    # Fall back to an empty repo if Raylib headers are unavailable.
    build = [
        "cc_library(",
        "    name = \"raylib\",",
        "    hdrs = [],",
        "    includes = [],",
        "    linkopts = [],",
        "    visibility = [\"//visibility:public\"],",
        ")",
        "",
    ]

    if prefix:
        # Symlink headers into the repository for hermetic includes
        ctx.symlink(prefix + "/include", "include")
        build = [
            "cc_library(",
            "    name = \"raylib\",",
            "    hdrs = glob([\"include/**/*.h\"]),",
            "    includes = [\"include\"],",
            "    linkopts = [",
            "        \"-L" + prefix + "/lib\",",
            "        \"-lraylib\",",
            "        \"-framework\", \"Cocoa\",",
            "        \"-framework\", \"IOKit\",",
            "        \"-framework\", \"CoreVideo\",",
            "        \"-framework\", \"OpenGL\",",
            "    ],",
            "    visibility = [\"//visibility:public\"],",
            ")",
            "",
        ]

    ctx.file("BUILD.bazel", "\n".join(build))

raylib_repo = repository_rule(
    implementation = _raylib_repo_impl,
    attrs = {
        "opt_homebrew": attr.string(default = "/opt/homebrew"),
        "usr_local": attr.string(default = "/usr/local"),
    },
)

def _raylib_ext_impl(module_ctx):
    # Always create the repo with sensible defaults; it will be empty if headers not present.
    raylib_repo(name = "raylib_homebrew")

raylib_ext = module_extension(
    implementation = _raylib_ext_impl,
)
