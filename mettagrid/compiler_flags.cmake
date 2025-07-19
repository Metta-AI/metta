# compiler_flags.cmake - Centralized compiler and linker flags configuration
# Prioritizes consistent warnings across GCC and Clang

# Create interface libraries for different flag categories
add_library(mettagrid_warnings INTERFACE)
add_library(mettagrid_sanitizers INTERFACE)
add_library(mettagrid_debug_flags INTERFACE)
add_library(mettagrid_base_flags INTERFACE)
add_library(mettagrid_coverage INTERFACE)

# Coverage option
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# ========================= BASE FLAGS =========================
target_compile_options(mettagrid_base_flags INTERFACE
  -fvisibility=hidden
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-omit-frame-pointer>
)

target_compile_definitions(mettagrid_base_flags INTERFACE
  "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
)

set_target_properties(mettagrid_base_flags PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE ON
)

# ========================= COVERAGE FLAGS =========================
if(ENABLE_COVERAGE)
  target_compile_options(mettagrid_coverage INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
      --coverage
      -fprofile-arcs
      -ftest-coverage
    >
  )

  target_link_options(mettagrid_coverage INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
      --coverage
    >
  )

  # Disable optimization for accurate coverage
  target_compile_options(mettagrid_coverage INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
      -O0
      -fno-inline
    >
  )
endif()

# ========================= WARNING FLAGS =========================
# Only include warnings that work consistently across both GCC and Clang
target_compile_options(mettagrid_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    # Basic warning sets
    -Wall
    -Wextra
    -Wpedantic

    # Type safety and conversions (consistent behavior)
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wold-style-cast

    # Memory and alignment
    -Wcast-align
    -Wcast-qual

    # C++ specific
    -Woverloaded-virtual
    -Wnon-virtual-dtor

    # Logic and control flow
    $<$<CXX_COMPILER_ID:GNU>:-Wshadow=compatible-local> # gcc shadowing warnings are very aggressive by default
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wshadow>
    -Wfloat-equal
  >
)

# ========================= OPTIONAL STRICT MODE =========================
# Create a separate target for platform-specific warnings
# This can be enabled selectively for deeper analysis
add_library(mettagrid_strict_warnings INTERFACE)
target_compile_options(mettagrid_strict_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    # Clang-specific useful warnings
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -Wthread-safety
      -Wimplicit-int-conversion
      -Wshorten-64-to-32
    >
    # GCC-specific useful warnings
    $<$<CXX_COMPILER_ID:GNU>:
      -Wduplicated-cond
      -Wduplicated-branches
      -Wlogical-op
      -Wuseless-cast
      -Wnull-dereference
    >
  >
)

# ========================= DEBUG-ONLY FLAGS =========================
# Additional warnings for debug builds - keep minimal for consistency
target_compile_options(mettagrid_debug_flags INTERFACE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    # Only warnings that behave consistently
    -Wstrict-overflow=5
    -Wfloat-conversion
  >
)

# Debug definitions
target_compile_definitions(mettagrid_debug_flags INTERFACE
  $<$<CONFIG:Debug>:
    _GLIBCXX_DEBUG
    METTAGRID_DEBUG_ASSERTIONS
  >
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang,AppleClang>>:_LIBCPP_DEBUG=1>
  $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)


# ========================= SANITIZERS =========================
# Sanitizer compile flags - keep only cross-platform sanitizers
target_compile_options(mettagrid_sanitizers INTERFACE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    -fsanitize=address
    -fsanitize=undefined
    -fsanitize=float-divide-by-zero
    -fno-sanitize-recover=all
    -fstack-protector-strong

    # Disable problematic checks for compatibility
    -fno-sanitize=shift-base
    -fno-sanitize=shift-exponent

    # Platform-specific sanitizers moved to opt-in mode
  >
)

# Sanitizer link flags
target_link_options(mettagrid_sanitizers INTERFACE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    -fsanitize=address
    -fsanitize=undefined
  >
)

# ========================= CONVENIENCE TARGETS =========================
# Combined targets for easy use
add_library(mettagrid_common_flags INTERFACE)
target_link_libraries(mettagrid_common_flags INTERFACE
  mettagrid_base_flags
  mettagrid_warnings
  mettagrid_debug_flags
)

add_library(mettagrid_all_flags INTERFACE)
target_link_libraries(mettagrid_all_flags INTERFACE
  mettagrid_common_flags
  mettagrid_sanitizers
)

# Add coverage to all flags when enabled
if(ENABLE_COVERAGE)
  target_link_libraries(mettagrid_all_flags INTERFACE
    mettagrid_coverage
  )
endif()

# ========================= HELPER FUNCTION =========================
# Function to apply flags to a target
function(mettagrid_apply_flags target)
  cmake_parse_arguments(ARG "SANITIZERS;STRICT;COVERAGE" "" "" ${ARGN})

  target_link_libraries(${target} PRIVATE mettagrid_common_flags)

  if(ARG_SANITIZERS)
    target_link_libraries(${target} PRIVATE mettagrid_sanitizers)
  endif()

  if(ARG_STRICT)
    target_link_libraries(${target} PRIVATE mettagrid_strict_warnings)
  endif()

  if(ARG_COVERAGE OR ENABLE_COVERAGE)
    target_link_libraries(${target} PRIVATE mettagrid_coverage)
  endif()
endfunction()
