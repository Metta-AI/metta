
# Our files
{.passC: "-I/Users/me/p/metta/packages/mettagrid/cpp/include".}
{.passC: "-I/Users/me/p/metta/packages/mettagrid/cpp/include/mettagrid".}

# UV files
# uv run python -c "import pybind11,sys;print(pybind11.get_include())"
{.passC: "-I/Users/me/p/metta/.venv/lib/python3.11/site-packages/pybind11/include".}
{.passC: "-I/Users/me/p/metta/.venv/lib/python3.11/site-packages/numpy/_core/include".}

# brew files
{.passC: "-I/opt/homebrew/opt/llvm/include/c++/v1".}
{.passC: "-I/Users/me/.local/share/uv/python/cpython-3.11.7-macos-aarch64-none/include/python3.11".}
{.passL: "-L/Users/me/.local/share/uv/python/cpython-3.11.7-macos-aarch64-none/lib".}

# Flags
{.passL: "-lpython3.11".}
{.passC: "-std=gnu++20".}

# Compile
{.compile: "bindings/mettagrid_c.cpp".}
