cmake_minimum_required(VERSION 3.15...3.30)
project(mettagrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

option(BUILD_TESTS "Build C++ unit tests and benchmarks" OFF)

# ========================= METTAGRID =========================

# Force CMake to use virtual environment Python first
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)

set(PYBIND11_FINDPYTHON ON)
execute_process(
  COMMAND "python" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)
find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)
set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})

file(GLOB_RECURSE METTAGRID_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.hpp
)

message(STATUS "Found sources: ${METTAGRID_SOURCES}")

add_library(mettagrid_obj OBJECT ${METTAGRID_SOURCES})

target_link_libraries(mettagrid_obj PUBLIC pybind11::pybind11 Python3::Python)

target_include_directories(mettagrid_obj PUBLIC
  ${NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid
)

target_compile_definitions(mettagrid_obj PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

# ========================= COMPILER FLAGS =========================

# Base visibility flag
target_compile_options(mettagrid_obj PUBLIC "-fvisibility=hidden")

# Always-on safety flags (minimal performance impact)
target_compile_options(mettagrid_obj PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -Wall
    -Wextra
    -Wuninitialized
    -Winit-self
    -Wshadow
    -Wfloat-equal
    -Wnull-dereference
    -Wformat=2
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wcast-qual
    -Wold-style-cast
    -Wpedantic
    -Woverloaded-virtual
    -Wnon-virtual-dtor
    -Wdouble-promotion
    # Clang-specific useful warnings
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -Wthread-safety
      -Wimplicit-fallthrough
    >
    # GCC-specific useful warnings
    $<$<CXX_COMPILER_ID:GNU>:
      -Wmaybe-uninitialized
      -Wduplicated-cond
      -Wduplicated-branches
      -Wlogical-op
      -Wuseless-cast
    >
  >
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    /permissive-
    /w14640  # thread unsafe static member init
    /w14826  # conversion from 'type1' to 'type2' is sign-extended
    /w14905  # wide string literal cast to 'LPSTR'
    /w14906  # string literal cast to 'LPWSTR'
  >
)

# Frame pointers for better debugging (minimal perf impact)
target_compile_options(mettagrid_obj PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-omit-frame-pointer>
)

# Debug-only additional warnings
target_compile_options(mettagrid_obj PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    # Additional warnings for debug builds
    -Wstrict-overflow=5
    -Wundef
    -Wredundant-decls
    -Wwrite-strings
    -Wpointer-arith
    -Wdate-time
    -Wmissing-declarations
    -Wfloat-conversion
    -Wshift-overflow=2

    # Lifetime and memory warnings (Clang)
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -Wdangling
      -Wdangling-gsl
      -Wreturn-stack-address
      -Wloop-analysis
      -Wstring-conversion
      -Wconditional-uninitialized
      -Wsometimes-uninitialized
      -Wstatic-in-inline
      -Wunreachable-code-aggressive
      -Wthread-safety-analysis
      -Wthread-safety-beta
      -Wshorten-64-to-32  # Especially important on 64-bit systems
    >

    # GCC-specific debug warnings
    $<$<CXX_COMPILER_ID:GNU>:
      -Wstack-usage=8192  # Warn on functions using more than 8KB stack
      -Wtrampolines
      -Wvector-operation-performance
      -Wdisabled-optimization
      -Waggressive-loop-optimizations
      -Wstringop-truncation
      -Wformat-truncation=2
      -Wformat-overflow=2
      -Wstringop-overflow=4
      -Warray-bounds=2
      -Wrestrict
      -Wattribute-alias=2
    >
  >
)

# Debug-only runtime sanitizers
target_compile_options(mettagrid_obj PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    -fsanitize=address
    -fsanitize=undefined
    -fsanitize=float-divide-by-zero
    -fsanitize=float-cast-overflow
    -fno-sanitize-recover=all
    -fstack-protector-strong
    # Disable problematic checks that cause false positives in macOS STL
    -fno-sanitize=shift-base
    -fno-sanitize=shift-exponent
    # Clang-specific sanitizers
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -fsanitize=nullability
      -fsanitize=integer
      -fsanitize=implicit-conversion
      -fsanitize=local-bounds
      # Additional exclusions for macOS STL compatibility
      -fno-sanitize=unsigned-shift-base
    >
  >
)

# Debug-only linker flags for sanitizers
target_link_options(mettagrid_obj PUBLIC
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    -fsanitize=address
    -fsanitize=undefined
  >
)

# Debug-only definitions
target_compile_definitions(mettagrid_obj PRIVATE
  $<$<CONFIG:Debug>:
    _GLIBCXX_DEBUG  # STL debug mode (GCC)
    _LIBCPP_DEBUG=1  # STL debug mode (Clang)
    METTAGRID_DEBUG_ASSERTIONS  # Your own debug assertions
  >
)

# FORTIFY_SOURCE is incompatible with AddressSanitizer, so only add it for Release builds
target_compile_definitions(mettagrid_obj PRIVATE
  $<$<CONFIG:Release>:
    _FORTIFY_SOURCE=2  # Runtime buffer overflow detection (Release only)
  >
)

# ========================= PYTHON MODULE =========================

pybind11_add_module(mettagrid_c $<TARGET_OBJECTS:mettagrid_obj>)

# Apply sanitizer linker flags to the Python module in debug builds
target_link_options(mettagrid_c PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    -fsanitize=address
    -fsanitize=undefined
  >
)

install(TARGETS mettagrid_c DESTINATION "metta/mettagrid")

# ========================= TESTS =========================

if(BUILD_TESTS)
  include(${CMAKE_CURRENT_LIST_DIR}/tests.cmake)
endif()
