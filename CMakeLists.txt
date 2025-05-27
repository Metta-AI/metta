cmake_minimum_required(VERSION 3.15...3.30)
project(metta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)

set(pybind11_DIR ".venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_gae "metta/rl/fast_gae.cpp")
install(TARGETS fast_gae DESTINATION "metta/rl")
