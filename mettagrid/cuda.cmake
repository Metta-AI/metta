# cuda.cmake - Optional CUDA configuration for GPU-accelerated behavioral analysis

# ========================= CUDA DETECTION =========================
# Option to enable CUDA behavioral analysis
option(BUILD_WITH_CUDA "Build GPU-accelerated behavioral analysis (requires NVIDIA GPU)" OFF)

# Auto-detect CUDA on Linux/Windows, but not macOS
if(NOT DEFINED BUILD_WITH_CUDA)
  if(APPLE)
    set(BUILD_WITH_CUDA OFF)
    message(STATUS "CUDA is not supported on macOS - disabling GPU acceleration")
  else()
    # Try to find CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(BUILD_WITH_CUDA ON)
      message(STATUS "CUDA found - enabling GPU acceleration")
    else()
      set(BUILD_WITH_CUDA OFF)
      message(STATUS "CUDA not found - disabling GPU acceleration")
    endif()
  endif()
endif()

# If CUDA is not requested or not available, return early
if(NOT BUILD_WITH_CUDA)
  # Define stub to indicate CUDA is not available
  add_compile_definitions(CUDA_BEHAVIORAL_ANALYSIS_DISABLED)
  return()
endif()

# ========================= CUDA SETUP =========================
# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA compiler not found. Disabling GPU behavioral analysis.")
  set(BUILD_WITH_CUDA OFF CACHE BOOL "" FORCE)
  add_compile_definitions(CUDA_BEHAVIORAL_ANALYSIS_DISABLED)
  return()
endif()

# Enable CUDA language
enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Require CUDA 11.0+ for modern features
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "CUDA 11.0 or newer is required. Found version ${CUDAToolkit_VERSION}")
endif()

# ========================= CUDA ARCHITECTURE =========================
# Auto-detect GPU architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Try to detect installed GPUs
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_COMPUTE_CAPS
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(GPU_COMPUTE_CAPS)
    # Convert compute capabilities to architectures (e.g., "7.5" -> "75")
    string(REPLACE "." "" GPU_COMPUTE_CAPS "${GPU_COMPUTE_CAPS}")
    string(REPLACE "\n" ";" GPU_COMPUTE_CAPS "${GPU_COMPUTE_CAPS}")
    list(REMOVE_DUPLICATES GPU_COMPUTE_CAPS)
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_COMPUTE_CAPS})
    message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    # Fallback to common architectures for CUDA 11+
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()

# ========================= CUDA STANDARDS =========================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ========================= CUDA FLAGS =========================
# Create interface library for CUDA flags
add_library(mettagrid_cuda_flags INTERFACE)

# CUDA compile options
target_compile_options(mettagrid_cuda_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo                      # Add line info for profiling
    --expt-relaxed-constexpr      # Allow more constexpr usage
    --expt-extended-lambda        # Extended lambda support
    -use_fast_math                # Fast math operations

    # Optimization flags
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-G -g>      # Device debug info

    # Warning flags for CUDA code
    -Werror=cross-execution-space-call
    -Wno-deprecated-gpu-targets
  >
)

# CUDA-specific definitions
target_compile_definitions(mettagrid_cuda_flags INTERFACE
  CUDA_BEHAVIORAL_ANALYSIS_ENABLED
  $<$<CONFIG:Debug>:CUDA_DEBUG>
)

# ========================= CUDA SOURCES =========================
# Find CUDA source files - look in mettagrid directory
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.cuh
)

if(CUDA_SOURCES)
  # Add CUDA sources to the main source list
  set(METTAGRID_SOURCES ${METTAGRID_SOURCES} ${CUDA_SOURCES} PARENT_SCOPE)
  message(STATUS "Found CUDA sources: ${CUDA_SOURCES}")
endif()

# ========================= CUDA LIBRARIES =========================
# Create interface library for CUDA runtime dependencies
add_library(mettagrid_cuda_runtime INTERFACE)

target_link_libraries(mettagrid_cuda_runtime INTERFACE
  CUDA::cudart
  CUDA::cuda_driver
  $<$<VERSION_GREATER_EQUAL:${CUDAToolkit_VERSION},11.0>:CUDA::nvtx3>
)

# Function to apply CUDA settings to a target
function(configure_cuda_target target)
  target_link_libraries(${target} PUBLIC
    mettagrid_cuda_flags
    mettagrid_cuda_runtime
  )

  # Set CUDA-specific properties
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
  )

  # Include CUDA directories
  target_include_directories(${target} PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
endfunction()

# ========================= OPENMP (Optional) =========================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found - enabling CPU parallelization for behavioral analysis")
else()
  message(STATUS "OpenMP not found - CPU parallelization disabled")
endif()

# ========================= STATUS MESSAGES =========================
message(STATUS "CUDA Configuration:")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")

# Print GPU information
execute_process(
  COMMAND nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  OUTPUT_VARIABLE GPU_INFO
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GPU_INFO)
  message(STATUS "  Available GPUs:")
  string(REPLACE "\n" "\n    " GPU_INFO_FORMATTED "    ${GPU_INFO}")
  message(STATUS "${GPU_INFO_FORMATTED}")
endif()

# Set parent scope variables to indicate CUDA is available
set(METTAGRID_CUDA_AVAILABLE TRUE PARENT_SCOPE)
set(METTAGRID_CUDA_VERSION ${CUDAToolkit_VERSION} PARENT_SCOPE)
