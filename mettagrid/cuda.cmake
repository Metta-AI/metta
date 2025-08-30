# cuda.cmake - CUDA configuration with CPU fallback

# Option to explicitly disable CUDA
option(BUILD_WITH_CUDA "Build with CUDA support for behavioral analysis" ON)

# Initialize CUDA variables
set(METTAGRID_CUDA_AVAILABLE FALSE)
set(METTAGRID_CUDA_VERSION "")

# Only look for CUDA if not explicitly disabled and not on Apple
if(BUILD_WITH_CUDA AND NOT APPLE)
  # Check for CUDA toolkit
  find_package(CUDAToolkit QUIET)

  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")

    # Enable CUDA language
    enable_language(CUDA)

    # Set CUDA flags
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architectures - support common GPUs
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()

    # Create CUDA runtime interface library
    add_library(mettagrid_cuda_runtime INTERFACE)
    target_link_libraries(mettagrid_cuda_runtime INTERFACE CUDA::cudart)

    # Set variables for behavioral analysis
    set(METTAGRID_CUDA_AVAILABLE TRUE)
    set(METTAGRID_CUDA_VERSION ${CUDAToolkit_VERSION})

    # Add CUDA source files
    set(METTAGRID_SOURCES ${METTAGRID_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/matrix_profile.cu
    )

    message(STATUS "Behavioral analysis: GPU-accelerated (CUDA ${CUDAToolkit_VERSION})")
  else()
    message(STATUS "CUDA Toolkit not found - using CPU fallback for behavioral analysis")
  endif()
elseif(APPLE)
  message(STATUS "CUDA not supported on macOS - using CPU fallback for behavioral analysis")
else()
  message(STATUS "CUDA disabled by user - using CPU fallback for behavioral analysis")
endif()

# If CUDA is not available, add CPU implementation and define macro
if(NOT METTAGRID_CUDA_AVAILABLE)
  # Add CPU implementation source
  set(METTAGRID_SOURCES ${METTAGRID_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/matrix_profile_cpu.cpp
  )

  # Check for OpenMP for CPU parallelization
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - CPU behavioral analysis will use parallel processing")
  else()
    message(STATUS "OpenMP not found - CPU behavioral analysis will be single-threaded")
  endif()
endif()

# Create an interface library for CUDA configuration
add_library(mettagrid_cuda_config INTERFACE)

# Set compile definitions based on CUDA availability
if(NOT METTAGRID_CUDA_AVAILABLE)
  target_compile_definitions(mettagrid_cuda_config INTERFACE CUDA_DISABLED)
endif()

# Function to configure CUDA properties for a target
function(configure_cuda_target target)
  # Always link the configuration
  target_link_libraries(${target} PUBLIC mettagrid_cuda_config)

  if(METTAGRID_CUDA_AVAILABLE)
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # CUDA compile options
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -fPIC
      >
    )

    # Link CUDA runtime
    target_link_libraries(${target} PUBLIC mettagrid_cuda_runtime)
  endif()
endfunction()

# Summary function
function(print_cuda_summary)
  message(STATUS "")
  message(STATUS "Behavioral Analysis Configuration:")
  if(METTAGRID_CUDA_AVAILABLE)
    message(STATUS "  Mode: GPU-accelerated")
    message(STATUS "  CUDA Version: ${METTAGRID_CUDA_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "  Mode: CPU fallback")
    if(OpenMP_CXX_FOUND)
      message(STATUS "  CPU Parallelization: OpenMP enabled")
    else()
      message(STATUS "  CPU Parallelization: Single-threaded")
    endif()
    if(APPLE)
      message(STATUS "  Reason: macOS platform (CUDA not supported)")
    elseif(NOT BUILD_WITH_CUDA)
      message(STATUS "  Reason: CUDA disabled by user")
    else()
      message(STATUS "  Reason: CUDA Toolkit not found")
    endif()
  endif()
  message(STATUS "")
endfunction()
