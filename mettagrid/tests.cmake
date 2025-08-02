# tests.cmake - Test and benchmark configuration

# ========================= TEST DEPENDENCIES =========================
if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(BUILD_BENCHMARKS)
  # Disable building Benchmark's own tests
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
  )
  FetchContent_MakeAvailable(googlebenchmark)
endif()

# ========================= COMMON TEST FLAGS =========================
# Create interface library for test/benchmark warning suppressions
add_library(mettagrid_test_suppressions INTERFACE)

target_compile_options(mettagrid_test_suppressions INTERFACE
  # Suppress warnings that are expected in test/benchmark code
  $<$<CXX_COMPILER_ID:Clang,AppleClang>:
    -Wno-global-constructors      # TEST() and BENCHMARK() macros
    -Wno-exit-time-destructors    # Static test objects
    -Wno-weak-vtables            # Test fixtures
    -Wno-padded                  # Test structures
  >
  $<$<CXX_COMPILER_ID:GNU>:
    -Wno-effc++                  # Too strict for test code
  >
)

# ========================= PYTHON ENVIRONMENT =========================
# Get Python environment for test execution
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.base_prefix)"
  OUTPUT_VARIABLE PYTHONHOME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ========================= UNIT TESTS =========================
if(BUILD_TESTS)
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

  foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source} $<TARGET_OBJECTS:mettagrid_obj>)

    target_include_directories(${test_name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid"
      ${NUMPY_INCLUDE_DIR}
    )

    target_link_libraries(${test_name} PRIVATE
      pybind11::pybind11
      Python3::Python
      GTest::gtest
      GTest::gtest_main
      mettagrid_all_flags           # Full flags including sanitizers
      mettagrid_test_suppressions   # Warning suppressions
      mettagrid_cuda_config         # CUDA configuration (includes CUDA_DISABLED macro if needed)
    )

    # If CUDA is available, configure CUDA properties
    if(METTAGRID_CUDA_AVAILABLE)
      configure_cuda_target(${test_name})
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name} --gtest_color=yes)

    # Set test environment with sanitizer options for Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      # Configure sanitizer options based on platform
      # Note: detect_leaks is not supported on macOS
      if(APPLE)
        set(ASAN_OPTIONS_VALUE "check_initialization_order=1:strict_init_order=1")
      else()
        set(ASAN_OPTIONS_VALUE "detect_leaks=1:check_initialization_order=1:strict_init_order=1")
      endif()

      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "PYTHONHOME=${PYTHONHOME};PYTHONPATH=${PYTHON_SITE_PACKAGES};ASAN_OPTIONS=${ASAN_OPTIONS_VALUE};UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
        LABELS "test"
      )
    else()
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "PYTHONHOME=${PYTHONHOME};PYTHONPATH=${PYTHON_SITE_PACKAGES}"
        LABELS "test"
      )
    endif()
  endforeach()
endif()

# ========================= BENCHMARKS =========================
if(BUILD_BENCHMARKS)
  file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp")

  foreach(bench_source ${BENCHMARK_SOURCES})
    get_filename_component(bench_name ${bench_source} NAME_WE)

    add_executable(${bench_name} ${bench_source} $<TARGET_OBJECTS:mettagrid_obj>)

    target_include_directories(${bench_name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid"
      ${NUMPY_INCLUDE_DIR}
    )

    target_link_libraries(${bench_name} PRIVATE
      pybind11::pybind11
      Python3::Python
      benchmark::benchmark
      benchmark::benchmark_main
      mettagrid_common_flags        # Base flags WITHOUT sanitizers
      mettagrid_test_suppressions   # Warning suppressions
      mettagrid_cuda_config         # CUDA configuration
    )

    # If CUDA is available, configure CUDA properties
    if(METTAGRID_CUDA_AVAILABLE)
      configure_cuda_target(${bench_name})
    endif()

    # Explicitly disable sanitizers for benchmarks in all builds
    target_compile_options(${bench_name} PRIVATE
      -fno-sanitize=address
      -fno-sanitize=undefined
    )
    target_link_options(${bench_name} PRIVATE
      -fno-sanitize=address
      -fno-sanitize=undefined
    )

    # Create wrapper for pretty output
    set(wrapper_script "${CMAKE_BINARY_DIR}/run_${bench_name}.cmake")
    file(WRITE ${wrapper_script} "
message(\"=================================================================================\")
message(\"BENCHMARK: ${bench_name}\")
message(\"=================================================================================\")
message(STATUS \"Running ${CMAKE_CURRENT_BINARY_DIR}/${bench_name}\")
execute_process(
  COMMAND \"${CMAKE_CURRENT_BINARY_DIR}/${bench_name}\"
  RESULT_VARIABLE result
)
if(result)
  message(FATAL_ERROR \"Benchmark failed with result: \${result}\")
endif()
")

    add_test(NAME ${bench_name}
      COMMAND ${CMAKE_COMMAND} -P ${wrapper_script}
    )
    set_tests_properties(${bench_name} PROPERTIES
      ENVIRONMENT "PYTHONHOME=${PYTHONHOME};PYTHONPATH=${PYTHON_SITE_PACKAGES}"
      LABELS "benchmark"
      TIMEOUT 300
    )
  endforeach()
endif()
