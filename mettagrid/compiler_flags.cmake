# compiler_flags.cmake - Centralized compiler and linker flags configuration

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
# Core warnings (Wall and Wextra cover a lot)
# Flags NOT in Wall/Wextra that add value:
target_compile_options(mettagrid_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -Wall
    -Wextra
    -Wpedantic
    # Type safety and conversions
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
    -Wshadow
    -Wfloat-equal
    -Wnull-dereference
    # Clang-specific useful warnings
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -Wthread-safety
    >
    # GCC-specific useful warnings
    $<$<CXX_COMPILER_ID:GNU>:
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

# ========================= DEBUG-ONLY FLAGS =========================
# Additional warnings for debug builds
target_compile_options(mettagrid_debug_flags INTERFACE
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
    # Data flow analysis
    -Wstrict-overflow=5
    -Wfloat-conversion
    # Additional Clang analysis
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
      -Wdangling
      -Wreturn-stack-address
      -Wloop-analysis
      -Wconditional-uninitialized
      -Wthread-safety-beta
      -Wshorten-64-to-32
    >
    # GCC-specific debug warnings
    $<$<CXX_COMPILER_ID:GNU>:
      -Wstack-usage=8192
      -Wstringop-truncation
      -Wformat-truncation=2
      -Wformat-overflow=2
      -Wstringop-overflow=4
      -Warray-bounds=2
    >
  >
)

# Debug definitions
target_compile_definitions(mettagrid_debug_flags INTERFACE
  $<$<CONFIG:Debug>:
    _GLIBCXX_DEBUG  # STL debug mode (GCC)
    _LIBCPP_DEBUG=1  # STL debug mode (Clang)
    METTAGRID_DEBUG_ASSERTIONS  # Your own debug assertions
  >
  # FORTIFY_SOURCE for Release builds only (incompatible with ASan)
  $<$<CONFIG:Release>:
    _FORTIFY_SOURCE=2  # Runtime buffer overflow detection
  >
)

# ========================= SANITIZERS =========================
# Sanitizer compile flags
target_compile_options(mettagrid_sanitizers INTERFACE
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
