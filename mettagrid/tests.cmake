# ========================= DEPENDENCIES =========================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)

FetchContent_MakeAvailable(googletest)

# Disable building Benchmark's own tests
set(BENCHMARK_ENABLE_TESTING
  OFF
  CACHE BOOL "" FORCE)

set(BENCHMARK_ENABLE_GTEST_TESTS
  OFF
  CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)

FetchContent_MakeAvailable(googlebenchmark)

# # ========================= TESTS =========================

# Enable CTest
enable_testing()

# Get Python base prefix for proper runtime configuration
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.base_prefix)"
  OUTPUT_VARIABLE PYTHONHOME
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get Python site-packages directory
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Helper function to build tests and benchmarks.
function(mettagrid_add_tests GLOB_PATTERN # e.g. "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        LINK_LIBS # semicolon-separated list of target names
        TEST_TYPE # "test" or "benchmark"
)
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${GLOB_PATTERN})

  foreach(src IN LISTS sources)
    get_filename_component(output_name ${src} NAME_WE)
    get_filename_component(src_dir ${src} DIRECTORY)

    # Mirror source subdir under build tree.
    # Note: this is a precaution against name collisions under `tests/` and `benchmarks/` dirs.
    # But it's not enough since we still can get duplicate target names in cmake
    # (since they don't include the full dir path).
    file(RELATIVE_PATH rel_dir "${CMAKE_SOURCE_DIR}" "${src_dir}")
    set(output_dir "${CMAKE_BINARY_DIR}/${rel_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    add_executable(${output_name} ${src} $<TARGET_OBJECTS:mettagrid_obj>)

    target_include_directories(${output_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/metta/mettagrid" ${NUMPY_INCLUDE_DIR})

    target_link_libraries(${output_name} PRIVATE ${LINK_LIBS})

    # Apply all flags (including sanitizers) to test executables
    target_link_libraries(${output_name} PRIVATE mettagrid_all_flags)

    set_target_properties(${output_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${output_dir}")

    if(TEST_TYPE STREQUAL "test")
      add_test(NAME ${output_name} COMMAND ${output_name} --gtest_color=yes)
      set_tests_properties(${output_name} PROPERTIES
        ENVIRONMENT "PYTHONHOME=${PYTHONHOME};PYTHONPATH=${PYTHON_SITE_PACKAGES}"
        LABELS "test")
    elseif(TEST_TYPE STREQUAL "benchmark")
      # Create a wrapper script for pretty benchmark output
      set(wrapper_script "${CMAKE_BINARY_DIR}/run_${output_name}.cmake")
      file(WRITE ${wrapper_script} "
message(\"\\n=================================================================================\")
message(\"BENCHMARK: ${output_name}\")
message(\"=================================================================================\")
execute_process(
  COMMAND \"${output_dir}/${output_name}\"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE
)
if(result)
  message(FATAL_ERROR \"Benchmark failed with result: \${result}\")
endif()
message(\"================================================================================\\n\")
")

      add_test(NAME ${output_name} COMMAND ${CMAKE_COMMAND} -P ${wrapper_script})
      set_tests_properties(${output_name} PROPERTIES
        ENVIRONMENT "PYTHONHOME=${PYTHONHOME};PYTHONPATH=${PYTHON_SITE_PACKAGES}"
        LABELS "benchmark"
        TIMEOUT 300)
    else()
      message(FATAL_ERROR "Invalid test type: ${TEST_TYPE}")
    endif()
  endforeach()
endfunction()

# Build tests
mettagrid_add_tests("${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
                      "pybind11::pybind11;Python3::Python;GTest::gtest;GTest::gtest_main" "test")

# Build benchmarks
mettagrid_add_tests("${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
                      "pybind11::pybind11;Python3::Python;benchmark::benchmark;benchmark::benchmark_main" "benchmark")
