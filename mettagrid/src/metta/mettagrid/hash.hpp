#ifndef HASH_HPP_
#define HASH_HPP_

#include <cstdint>
#include <string>

#include "rapidhash/rapidhash.h"

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

#endif  // HASH_HPP_
