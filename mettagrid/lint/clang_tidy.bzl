# Simple clang-tidy integration for Bazel
# Provides Bazel-native integration for clang-tidy linting

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _clang_tidy_aspect_impl(target, ctx):
    """Aspect to run clang-tidy on C++ targets."""
    
    # Only process C++ rules
    if not ctx.rule.kind in ["cc_library", "cc_binary", "cc_test", "pybind_extension", "pybind_library"]:
        return []
    
    # Skip if no CcInfo provider
    if not CcInfo in target:
        return []
    
    cc_info = target[CcInfo]
    if not cc_info.compilation_context:
        return []
    
    # Get the C++ toolchain
    cc_toolchain = find_cpp_toolchain(ctx)
    
    # Collect source files
    srcs = []
    for src in ctx.rule.files.srcs:
        if src.extension in ["cpp", "cc", "cxx", "c"]:
            srcs.append(src)
    
    if not srcs:
        return []
    
    compilation_context = cc_info.compilation_context
    
    # Create report files
    reports = []
    for src in srcs:
        # Make output path unique per target to avoid conflicts
        report = ctx.actions.declare_file(
            "_clang_tidy_/" + ctx.label.name + "/" + src.basename + ".report",
        )
        reports.append(report)
        
        # Build the command
        cmd = []
        cmd.append(ctx.executable._clang_tidy.path)
        cmd.append(src.path)
        
        # Add config file if present
        if ctx.file._clang_tidy_config:
            cmd.append("--config-file=" + ctx.file._clang_tidy_config.path)
        
        # Add compilation flags
        cmd.append("--")
        cmd.append("-std=c++20")
        
        # Include paths
        for include in compilation_context.includes.to_list():
            cmd.append("-I" + include)
        
        for include in compilation_context.quote_includes.to_list():
            cmd.extend(["-iquote", include])
        
        for include in compilation_context.system_includes.to_list():
            cmd.extend(["-isystem", include])
        
        # Defines
        for define in compilation_context.defines.to_list():
            cmd.append("-D" + define)
        
        # Local defines
        if hasattr(compilation_context, "local_defines"):
            for define in compilation_context.local_defines.to_list():
                cmd.append("-D" + define)
        
        # Add Python and pybind11 specific flags
        cmd.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
        
        # Run clang-tidy and capture output
        ctx.actions.run_shell(
            inputs = depset(
                direct = [src] + ([ctx.file._clang_tidy_config] if ctx.file._clang_tidy_config else []),
                transitive = [compilation_context.headers],
            ),
            outputs = [report],
            command = "{} > {} 2>&1 || true".format(
                shell.array_literal(cmd),
                shell.quote(report.path),
            ),
            mnemonic = "ClangTidy",
            progress_message = "Running clang-tidy on %s" % src.short_path,
            tools = [ctx.executable._clang_tidy],
        )
    
    return [
        OutputGroupInfo(
            clang_tidy_reports = depset(reports),
        ),
    ]

clang_tidy_aspect = aspect(
    implementation = _clang_tidy_aspect_impl,
    attrs = {
        "_clang_tidy": attr.label(
            default = Label("//lint:clang_tidy_wrapper"),
            executable = True,
            cfg = "exec",
        ),
        "_clang_tidy_config": attr.label(
            default = Label("//:.clang-tidy"),
            allow_single_file = True,
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)

def _clang_tidy_test_impl(ctx):
    """Test rule that fails if clang-tidy reports errors."""
    
    reports = []
    for dep in ctx.attr.deps:
        if OutputGroupInfo in dep and hasattr(dep[OutputGroupInfo], "clang_tidy_reports"):
            reports.extend(dep[OutputGroupInfo].clang_tidy_reports.to_list())
    
    # Create a test script
    test_script = ctx.actions.declare_file(ctx.label.name + ".sh")
    
    if reports:
        # Create script that checks for errors
        script_lines = ["#!/bin/bash", "set -e", "ERRORS_FOUND=0", "", "echo 'Checking clang-tidy reports...'", ""]
        
        for report in reports:
            script_lines.extend([
                "if [ -f '{}' ]; then".format(report.short_path),
                "    if grep -q 'error:' '{}' 2>/dev/null; then".format(report.short_path),
                "        echo 'Errors found in {}:'".format(report.short_path),
                "        cat '{}'".format(report.short_path),
                "        ERRORS_FOUND=1",
                "    elif grep -q 'warning:' '{}' 2>/dev/null; then".format(report.short_path),
                "        echo 'Warnings found in {} (not failing):'".format(report.short_path),
                "        cat '{}'".format(report.short_path),
                "    fi",
                "fi",
                "",
            ])
        
        script_lines.extend([
            "if [ $ERRORS_FOUND -eq 0 ]; then",
            "    echo 'All files passed clang-tidy checks!'",
            "    exit 0",
            "else",
            "    echo 'clang-tidy found issues!'",
            "    exit 1",
            "fi",
        ])
        
        ctx.actions.write(test_script, "\n".join(script_lines), is_executable = True)
    else:
        # No files to check
        ctx.actions.write(
            test_script,
            "#!/bin/bash\necho 'No C++ files to check'\nexit 0\n",
            is_executable = True,
        )
    
    return [
        DefaultInfo(
            executable = test_script,
            runfiles = ctx.runfiles(files = reports),
        ),
    ]

clang_tidy_test = rule(
    implementation = _clang_tidy_test_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [clang_tidy_aspect],
            doc = "C++ targets to run clang-tidy on",
        ),
    },
    test = True,
)

def _clang_tidy_all_impl(ctx):
    """Rule to run clang-tidy and produce a summary report."""
    
    reports = []
    for dep in ctx.attr.deps:
        if OutputGroupInfo in dep and hasattr(dep[OutputGroupInfo], "clang_tidy_reports"):
            reports.extend(dep[OutputGroupInfo].clang_tidy_reports.to_list())
    
    # Create a summary report
    summary = ctx.actions.declare_file(ctx.label.name + "_summary.txt")
    
    if reports:
        script_lines = [
            "echo 'Clang-Tidy Report Summary' > {}".format(shell.quote(summary.path)),
            "echo '=========================' >> {}".format(shell.quote(summary.path)),
            "echo '' >> {}".format(shell.quote(summary.path)),
        ]
        
        for report in reports:
            script_lines.extend([
                "if [ -f '{}' ] && [ -s '{}' ]; then".format(report.path, report.path),
                "    echo 'File: {}' >> {}".format(report.short_path, shell.quote(summary.path)),
                "    cat '{}' >> {}".format(report.path, shell.quote(summary.path)),
                "    echo '' >> {}".format(shell.quote(summary.path)),
                "fi",
            ])
        
        ctx.actions.run_shell(
            inputs = reports,
            outputs = [summary],
            command = "\n".join(script_lines),
        )
    else:
        ctx.actions.write(summary, "No C++ files to lint.\n")
    
    return [DefaultInfo(files = depset([summary]))]

clang_tidy_all = rule(
    implementation = _clang_tidy_all_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [clang_tidy_aspect],
            doc = "C++ targets to run clang-tidy on",
        ),
    },
)