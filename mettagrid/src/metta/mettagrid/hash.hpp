#ifndef HASH_HPP_
#define HASH_HPP_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "rapidhash/rapidhash.h"
namespace py = pybind11;

// Why rapidhash instead of std::hash?
// - Performance: Even faster than wyhash (which was already 2-3x faster than std::hash)
// - Determinism: std::hash is not guaranteed to produce consistent results across
//   platforms, compilers, or program runs, making it unsuitable for reproducible
//   research where identical maps must produce identical hashes
// - Quality: Superior collision resistance and statistical distribution properties
// - Cross-platform consistency: Essential for ML research reproducibility where
//   training and evaluation may occur on different systems

inline uint64_t hash_string(const std::string& str, uint64_t seed = 0) {
  return rapidhash_withSeed(str.data(), str.size(), seed);
}

// Calculate a deterministic hash of a grid map from Python list structure
inline uint64_t hash_mettagrid_map(const py::list& map) {
  std::string grid_hash_data;

  int height = map.size();
  if (height > 0) {
    int width = map[0].cast<py::list>().size();

    // Pre-allocate for efficiency
    grid_hash_data.reserve(height * width * 20);

    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        auto py_cell = map[r].cast<py::list>()[c].cast<py::str>();
        auto cell = py_cell.cast<std::string>();

        // Add cell position and type to hash data
        grid_hash_data += std::to_string(r) + "," + std::to_string(c) + ":" + cell + ";";
      }
    }
  }

  return hash_string(grid_hash_data);
}

#endif  // HASH_HPP_
