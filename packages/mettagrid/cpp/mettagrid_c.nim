import os, strutils

# PWD is assumed to be the root of the mettagrid package.
# cd to metta/packages/mettagrid and run `nim cpp mettagrid_c.nim`
const pwd = getEnv("PWD")
{.passC: "-I" & pwd & "/include/mettagrid".}

# Flags
{.passL: "-lpython3.11".} # We only support Python 3.11.
{.passC: "-std=gnu++20".} # We only support C++20.

# Get Python paths from system
const pythonInclude = staticExec("python3 -c \"import sysconfig; print(sysconfig.get_path('include'))\"").strip()
when pythonInclude.len == 0:
  {.error: "python include path not found".}
else:
  {.passC: "-I" & pythonInclude.}

const pythonLibDir = staticExec("python3 -c \"import sysconfig; print(sysconfig.get_config_var('LIBDIR'))\"").strip()
when pythonLibDir.len == 0:
  {.error: "python lib dir not found".}
else:
  {.passL: "-L" & pythonLibDir.}

# Get pybind11 and numpy include paths
const pybind11Include = staticExec("python3 -c \"import pybind11; print(pybind11.get_include())\"").strip()
when pybind11Include.len == 0:
  {.error: "pybind11 include path not found".}
else:
  {.passC: "-I" & pybind11Include.}

const numpyInclude = staticExec("python3 -c \"import numpy; print(numpy.get_include())\"").strip()
when numpyInclude.len == 0:
  {.error: "numpy include path not found".}
else:
  {.passC: "-I" & numpyInclude.}

# Compile
{.compile: "bindings/mettagrid_c.cpp".}
