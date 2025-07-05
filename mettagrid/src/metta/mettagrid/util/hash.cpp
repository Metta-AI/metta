/**
 * hash.cpp - Python bindings for hash utilities
 *
 * This module exposes the hash functions used in MettaGrid.
 * Located at: metta/mettagrid/util/hash.cpp
 */

#include "hash.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(hash, m) {
  m.doc() = "Hash utilities for MettaGrid";

  m.def("hash_string",
        &hash_string,
        py::arg("str"),
        py::arg("seed") = 0,
        "Hash a string using rapidhash algorithm.\n\n"
        "Args:\n"
        "    str: The string to hash\n"
        "    seed: Optional seed value (default: 0)\n\n"
        "Returns:\n"
        "    64-bit hash value");

  m.def("hash_mettagrid_map",
        &hash_mettagrid_map,
        py::arg("map"),
        "Calculate a deterministic hash of a MettaGrid map.\n\n"
        "Args:\n"
        "    map: A list of lists of strings representing the grid\n\n"
        "Returns:\n"
        "    64-bit hash value representing the map configuration");

  m.attr("DEFAULT_SEED") = py::int_(0);
}
