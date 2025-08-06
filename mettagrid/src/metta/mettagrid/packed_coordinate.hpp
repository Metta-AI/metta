#ifndef PACKED_COORDINATE_HPP_
#define PACKED_COORDINATE_HPP_
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "objects/constants.hpp"

/**
 * Utilities for packing/unpacking grid coordinates into compact byte representation.
 *
 * Provides space-efficient coordinate storage for contexts where memory is at a premium
 * (e.g., observation tokens). This is a compressed alternative to GridLocation when
 * the layer component is not needed and coordinates fit in 4 bits each.
 *
 * Packing scheme:
 * - Upper 4 bits: row (r / y-coordinate)
 * - Lower 4 bits: col (c / x-coordinate)
 * - Special value 0xFF represents empty/invalid coordinate
 */
namespace PackedCoordinate {

// Constants for bit packing
constexpr uint8_t ROW_SHIFT = 4;
constexpr uint8_t COL_MASK = 0x0F;
constexpr uint8_t ROW_MASK = 0xF0;

// Maximum coordinate value that can be packed (4 bits = 0-14)
constexpr uint8_t MAX_PACKABLE_COORD = 14;

/**
 * Pack grid coordinates into a single byte.
 *
 * @param row Row coordinate (r in GridLocation, 0-14)
 * @param col Column coordinate (c in GridLocation, 0-14)
 * @return Packed coordinate byte
 * @note The value 0xFF is reserved to indicate 'empty'
 * @throws std::invalid_argument if row or col > 14
 */
inline uint8_t pack(uint8_t row, uint8_t col) {
  if (row > MAX_PACKABLE_COORD || col > MAX_PACKABLE_COORD) {
    throw std::invalid_argument("Coordinates must be <= " + std::to_string(MAX_PACKABLE_COORD) +
                                ". Got row=" + std::to_string(row) + ", col=" + std::to_string(col));
  }
  return static_cast<uint8_t>((row << ROW_SHIFT) | (col & COL_MASK));
}

/**
 * Unpack byte into coordinates with empty handling.
 *
 * @param packed Packed coordinate byte
 * @return std::optional<std::pair<row, col>> or std::nullopt if empty
 */
inline std::optional<std::pair<uint8_t, uint8_t>> unpack(uint8_t packed) {
  if (packed == EmptyTokenByte) {
    return std::nullopt;
  }
  uint8_t row = (packed & ROW_MASK) >> ROW_SHIFT;
  uint8_t col = packed & COL_MASK;
  return {{row, col}};
}
/**
 * Check if a packed coordinate represents an empty/invalid position.
 */
inline bool is_empty(uint8_t packed_data) {
  return packed_data == EmptyTokenByte;
}

struct ObservationPattern {
  int height, width;

  struct Iterator {
    int d = 0;     // Current manhattan distance
    int dr = 0;    // Current row offset
    int step = 0;  // Step for dc symmetry: 0 = -dc, 1 = +dc
    int emitted = 0;
    const int max_emitted;
    const int row_min, row_max, col_min, col_max;
    std::pair<int, int> current;
    bool done = false;

    Iterator(int h, int w, bool is_end = false)
        : max_emitted(h * w), row_min(-h / 2), row_max(h / 2), col_min(-w / 2), col_max(w / 2), current{0, 0} {
      if (is_end) {
        done = true;
      } else {
        advance();
      }
    }

    std::pair<int, int> operator*() const {
      return current;
    }

    Iterator& operator++() {
      advance();
      return *this;
    }

    bool operator!=(const Iterator& /*other*/) const {
      return !done;
    }

  private:
    void advance() {
      while (emitted < max_emitted) {
        while (dr <= d) {
          int dc = d - std::abs(dr);
          while (step < (dc == 0 ? 1 : 2)) {
            int r = dr;
            int c = (step == 0 ? -dc : dc);

            ++step;

            if (r >= row_min && r <= row_max && c >= col_min && c <= col_max) {
              current = {r, c};
              ++emitted;
              return;
            }
          }
          ++dr;
          step = 0;
        }
        ++d;
        dr = -d;  // ✅ reset for the next Manhattan shell: iterate from -d to +d
      }
      done = true;
    }
  };

  Iterator begin() const {
    return Iterator(height, width);
  }

  Iterator end() const {
    return Iterator(height, width, true);
  }
};

}  // namespace PackedCoordinate
#endif  // PACKED_COORDINATE_HPP_
