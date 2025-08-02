cmake_minimum_required(VERSION 3.18...3.30)  # 3.18+ for better CUDA support
project(mettagrid LANGUAGES CXX)  # CUDA will be enabled conditionally

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ========================= OPTIONS =========================
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# ========================= FETCHCONTENT CACHING =========================
# Configure FetchContent to use a shared directory for all builds
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH "FetchContent base directory")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE CACHE BOOL "Disable updates for FetchContent packages")

include(FetchContent)

# Include centralized compiler flags
include(${CMAKE_CURRENT_LIST_DIR}/compiler_flags.cmake)

# Include CUDA configuration (will auto-detect on Linux/Windows)
include(${CMAKE_CURRENT_LIST_DIR}/cuda.cmake)

# ========================= PYTHON SETUP =========================
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

# ========================= METTAGRID LIBRARY =========================
file(GLOB_RECURSE METTAGRID_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid/*.hpp
)

# CUDA sources are added by cuda.cmake if available
message(STATUS "Found sources: ${METTAGRID_SOURCES}")

add_library(mettagrid_obj OBJECT ${METTAGRID_SOURCES})
target_link_libraries(mettagrid_obj PUBLIC pybind11::pybind11 Python3::Python)
target_include_directories(mettagrid_obj PUBLIC
  ${NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid
)

# Apply all flags to the object library
target_link_libraries(mettagrid_obj PRIVATE mettagrid_all_flags)

# Link the CUDA configuration (this provides CUDA_DISABLED macro if needed)
target_link_libraries(mettagrid_obj PUBLIC mettagrid_cuda_config)

# Configure CUDA if available
if(METTAGRID_CUDA_AVAILABLE)
  configure_cuda_target(mettagrid_obj)
endif()

# Link OpenMP if found and CUDA is available
if(METTAGRID_CUDA_AVAILABLE AND OpenMP_CXX_FOUND)
  target_link_libraries(mettagrid_obj PUBLIC OpenMP::OpenMP_CXX)
endif()

# ========================= PYTHON MODULE =========================
pybind11_add_module(mettagrid_c $<TARGET_OBJECTS:mettagrid_obj>)

# Python module needs sanitizer link flags in debug builds
target_link_libraries(mettagrid_c PRIVATE mettagrid_sanitizers)

# Link CUDA configuration to ensure macros are available
target_link_libraries(mettagrid_c PRIVATE mettagrid_cuda_config)

# If CUDA is available, link CUDA runtime to Python module
if(METTAGRID_CUDA_AVAILABLE)
  target_link_libraries(mettagrid_c PRIVATE mettagrid_cuda_runtime)
endif()

install(TARGETS mettagrid_c DESTINATION "metta/mettagrid")

# ========================= TESTS & BENCHMARKS =========================
if(BUILD_TESTS OR BUILD_BENCHMARKS)
  enable_testing()
  include(${CMAKE_CURRENT_LIST_DIR}/tests.cmake)
endif()

# ========================= SUMMARY =========================
message(STATUS "")
message(STATUS "MettaGrid Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Enable Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  CUDA Support: ${BUILD_WITH_CUDA}")
if(METTAGRID_CUDA_AVAILABLE)
  message(STATUS "  CUDA Version: ${METTAGRID_CUDA_VERSION}")
  message(STATUS "  Behavioral Analysis: GPU")
else()
  message(STATUS "  Behavioral Analysis: CPU")
endif()
message(STATUS "")
