# ========================= DEPENDENCIES =========================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

# Disable building Benchmark's own tests
set(BENCHMARK_ENABLE_TESTING    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)

FetchContent_MakeAvailable(googlebenchmark)

# # ========================= TESTS =========================

# 1) Enable CTest
enable_testing()

# 2) Compile & register all tests
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

set(PYBIND11_TESTS test_mettagrid)

foreach(test_src IN LISTS TEST_SOURCES)
  get_filename_component(test_name ${test_src} NAME_WE)
  get_filename_component(test_src_dir ${test_src} DIRECTORY)

  add_executable(${test_name} ${test_src} $<TARGET_OBJECTS:mettagrid_obj>)

  target_include_directories(
    ${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mettagrid"
                         ${NUMPY_INCLUDE_DIR})

  # All tests need Python linking since they use mettagrid_obj which contains
  # pybind11 code
  target_link_libraries(${test_name} PRIVATE pybind11::pybind11 Python3::Python
                                             GTest::gtest GTest::gtest_main)

  # Pass Python base prefix for runtime configuration
  target_compile_definitions(${test_name}
                             PRIVATE PYTHON_BASE_PREFIX="${PYTHON_BASE_PREFIX}")

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
    ${bench_name} PRIVATE pybind11::pybind11 Python3::Python
                          benchmark::benchmark benchmark::benchmark_main)

  set_target_properties(${bench_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${bench_src_dir}")

endforeach()
