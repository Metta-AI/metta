cmake_minimum_required(VERSION 3.15...3.30)
project(mettagrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

option(BUILD_TESTS "Build C++ unit tests and benchmarks" OFF)

# ========================= METTAGRID =========================

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

target_compile_options(mettagrid_obj PUBLIC "-fvisibility=hidden")

pybind11_add_module(mettagrid_c $<TARGET_OBJECTS:mettagrid_obj>)

install(TARGETS mettagrid_c DESTINATION "metta/mettagrid")

# ========================= TESTS =========================

if(BUILD_TESTS)
  include(${CMAKE_CURRENT_LIST_DIR}/tests.cmake)
endif()
