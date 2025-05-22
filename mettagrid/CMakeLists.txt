cmake_minimum_required(VERSION 3.15...3.30)
project(mettagrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================= DEPENDENCIES =========================

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main # or pin to a release, e.g. v1.7.2
)

# Disable building Benchmark's own tests (optional)
set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# ========================= METTAGRID =========================

set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR
    "../.venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)
set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})

file(GLOB_RECURSE METTAGRID_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/mettagrid/*.cpp)

add_library(mettagrid_obj OBJECT ${METTAGRID_SOURCES} # mettagrid/*.cpp
)

# pull in pybind11’s own includes (so <pybind11/numpy.h> resolves)
target_link_libraries(mettagrid_obj PUBLIC pybind11::pybind11)

# pull in NumPy’s include dir (so the C-API macros are found)
target_include_directories(mettagrid_obj PUBLIC ${NUMPY_INCLUDE_DIR})

target_include_directories(mettagrid_obj
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/mettagrid")
target_compile_definitions(mettagrid_obj
                           PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_compile_options(mettagrid_obj PUBLIC -fvisibility=hidden)

pybind11_add_module(mettagrid_c $<TARGET_OBJECTS:mettagrid_obj>)

install(TARGETS mettagrid_c DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/mettagrid")

# # ========================= TESTS =========================

# 1) Enable CTest
enable_testing()

# 2) Compile & register all tests
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

set(PYBIND11_TESTS test_mettagrid)

foreach(test_src IN LISTS TEST_SOURCES BENCH_SOURCES)
  get_filename_component(test_name ${test_src} NAME_WE)
  get_filename_component(test_src_dir ${test_src} DIRECTORY)

  add_executable(${test_name} ${test_src} $<TARGET_OBJECTS:mettagrid_obj>)

  target_include_directories(
    ${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mettagrid"
                         ${NUMPY_INCLUDE_DIR})

  # pick link libraries based on whether this test is in our pybind11 list
  if(test_name IN_LIST PYBIND11_TESTS)
    message(STATUS "Configuring Python-backed test: ${test_name}")
    target_link_libraries(
      ${test_name} PRIVATE pybind11::pybind11 Python3::Python GTest::gtest
                           GTest::gtest_main)
  else()
    message(STATUS "Configuring pure C++ test: ${test_name}")
    target_link_libraries(${test_name} PRIVATE pybind11::pybind11 GTest::gtest GTest::gtest_main)
  endif()

  add_test(NAME ${test_name} COMMAND ${test_name} --gtest_color=yes)

  set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                "${test_src_dir}")
endforeach()

# 3) Compile all benchmarks under benchmarks/
file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp")

foreach(bench_src IN LISTS BENCH_SOURCES)

  get_filename_component(bench_name ${bench_src} NAME_WE)
  get_filename_component(bench_src_dir ${bench_src} DIRECTORY)

  add_executable(${bench_name} ${bench_src} $<TARGET_OBJECTS:mettagrid_obj>)

  target_include_directories(
    ${bench_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mettagrid"
                          ${NUMPY_INCLUDE_DIR})

  target_link_libraries(
    ${bench_name} PRIVATE pybind11::pybind11 benchmark::benchmark
                          benchmark::benchmark_main)

  set_target_properties(${bench_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${bench_src_dir}")

endforeach()

# 4) “make test” → ctest “make benchmark” → just builds benchmarks; you can run
# them manually, or add a wrapper target:

add_custom_target(
  run_benchmarks
  DEPENDS ${BENCH_SOURCES}
  COMMENT
    "Building all benchmarks. Run them with ./<bench_name> in build directory.")
