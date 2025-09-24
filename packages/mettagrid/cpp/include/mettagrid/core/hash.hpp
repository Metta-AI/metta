// wyhash_minimal.hpp - Minimal wyhash implementation for grid checksums
// Based on wyhash v4.2 by Wang Yi (王一)
// Original: https://github.com/wangyi-fudan/wyhash
// License: Public Domain (Unlicense)
//
// Why wyhash instead of std::hash?
// - Performance: 2-3x faster than many std::hash implementations
// - Determinism: std::hash is not guaranteed to produce consistent results across
//   platforms, compilers, or program runs, making it unsuitable for reproducible
//   research where identical maps must produce identical hashes
// - Quality: Superior collision resistance and statistical distribution properties
// - Cross-platform consistency: Essential for ML research reproducibility where
//   training and evaluation may occur on different systems

#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_HASH_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_HASH_HPP_

#include <cstdint>
#include <cstring>
#include <string>

namespace wyhash {

// Core constants
static constexpr uint64_t _wyp[4] = {0x2d358dccaa6c78a5ull,
                                     0x8bb84b93962eacc9ull,
                                     0x4b33a62ed433d4a3ull,
                                     0x4d5a2da51de1aa47ull};

// Platform-optimized multiply
inline void _wymum(uint64_t* A, uint64_t* B) {
#ifdef __SIZEOF_INT128__
  // Fast path for 64-bit platforms with 128-bit support
  __uint128_t r = *A;
  r *= *B;
  *A = static_cast<uint64_t>(r);
  *B = static_cast<uint64_t>(r >> 64);
#else
  // Portable fallback
  uint64_t ha = *A >> 32, hb = *B >> 32;
  uint64_t la = static_cast<uint32_t>(*A), lb = static_cast<uint32_t>(*B);
  uint64_t rh = ha * hb;
  uint64_t rm0 = ha * lb;
  uint64_t rm1 = hb * la;
  uint64_t rl = la * lb;
  uint64_t t = rl + (rm0 << 32);
  auto c = static_cast<uint64_t>(t < rl);
  uint64_t lo = t + (rm1 << 32);
  c += static_cast<uint64_t>(lo < t);
  uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
  *A = lo;
  *B = hi;
#endif
}

// Mix function
inline uint64_t _wymix(uint64_t A, uint64_t B) {
  _wymum(&A, &B);
  return A ^ B;
}

// Read functions with proper alignment handling
inline uint64_t _wyr8(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

inline uint64_t _wyr4(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline uint64_t _wyr3(const uint8_t* p, size_t k) {
  return ((static_cast<uint64_t>(p[0])) << 16) | ((static_cast<uint64_t>(p[k >> 1])) << 8) | p[k - 1];
}

// Main hash function
inline uint64_t hash(const void* key, size_t len, uint64_t seed = 0) {
  const uint8_t* p = static_cast<const uint8_t*>(key);
  seed ^= _wymix(seed ^ _wyp[0], _wyp[1]);
  uint64_t a, b;

  if (len <= 16) {
    if (len >= 4) {
      a = (_wyr4(p) << 32) | _wyr4(p + ((len >> 3) << 2));
      b = (_wyr4(p + len - 4) << 32) | _wyr4(p + len - 4 - ((len >> 3) << 2));
    } else if (len > 0) {
      a = _wyr3(p, len);
      b = 0;
    } else {
      a = b = 0;
    }
  } else {
    size_t i = len;
    if (i >= 48) {
      uint64_t see1 = seed, see2 = seed;
      do {
        seed = _wymix(_wyr8(p) ^ _wyp[1], _wyr8(p + 8) ^ seed);
        see1 = _wymix(_wyr8(p + 16) ^ _wyp[2], _wyr8(p + 24) ^ see1);
        see2 = _wymix(_wyr8(p + 32) ^ _wyp[3], _wyr8(p + 40) ^ see2);
        p += 48;
        i -= 48;
      } while (i >= 48);
      seed ^= see1 ^ see2;
    }
    while (i > 16) {
      seed = _wymix(_wyr8(p) ^ _wyp[1], _wyr8(p + 8) ^ seed);
      i -= 16;
      p += 16;
    }
    a = _wyr8(p + i - 16);
    b = _wyr8(p + i - 8);
  }

  a ^= _wyp[1];
  b ^= seed;
  _wymum(&a, &b);
  return _wymix(a ^ _wyp[0] ^ len, b ^ _wyp[1]);
}

// Convenience function for std::string
inline uint64_t hash_string(const std::string& str, uint64_t seed = 0) {
  return hash(str.data(), str.size(), seed);
}

}  // namespace wyhash

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_HASH_HPP_
