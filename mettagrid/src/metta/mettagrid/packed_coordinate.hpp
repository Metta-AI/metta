#ifndef PACKED_COORDINATE_HPP_
#define PACKED_COORDINATE_HPP_
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

// Single pre-computed pattern for maximum observable size (15x15)
struct ObservationPattern {
  static constexpr size_t MAX_OBSERVABLE_SIZE = MAX_PACKABLE_COORD + 1;
  static constexpr size_t MAX_PATTERN_SIZE = MAX_OBSERVABLE_SIZE * MAX_OBSERVABLE_SIZE;

  struct Offset {
    int8_t r_offset;
    int8_t c_offset;
  };

  // Pre-sorted offsets by Manhattan distance for the maximum 15x15 window
  std::array<Offset, MAX_PATTERN_SIZE> offsets;
  size_t size;  // Actual number of valid offsets

  ObservationPattern() {
    size = 0;
    const int8_t radius = MAX_OBSERVABLE_SIZE / 2;

    // Build list of all offsets with their distances
    struct OffsetWithDistance {
      int8_t r_offset;
      int8_t c_offset;
      uint8_t distance;
    };

    std::vector<OffsetWithDistance> temp_offsets;
    temp_offsets.reserve(MAX_PATTERN_SIZE);

    // Generate all positions in the maximum observation window
    for (int r = -radius; r <= radius; r++) {
      for (int c = -radius; c <= radius; c++) {
        uint8_t distance = std::abs(r) + std::abs(c);
        temp_offsets.push_back({static_cast<int8_t>(r), static_cast<int8_t>(c), distance});
      }
    }

    // Sort by Manhattan distance (stable sort preserves order for same distance)
    std::stable_sort(temp_offsets.begin(),
                     temp_offsets.end(),
                     [](const OffsetWithDistance& a, const OffsetWithDistance& b) { return a.distance < b.distance; });

    // Copy to our fixed array
    for (const auto& offset : temp_offsets) {
      offsets[size++] = {offset.r_offset, offset.c_offset};
    }
  }
};

// Get the global pattern instance
inline const ObservationPattern& get_observation_pattern() {
  static const ObservationPattern pattern;
  return pattern;
}

/**
 * Lightweight view into the observation pattern for a specific window size.
 * This class provides an iterator interface over the relevant offsets.
 */
class ObservationSearchPattern {
private:
  const ObservationPattern::Offset* begin_ptr;
  const ObservationPattern::Offset* end_ptr;

public:
  ObservationSearchPattern(uint8_t width, uint8_t height) {
    const auto& pattern = get_observation_pattern();

    // For a given observation window, we only need offsets within its bounds
    uint8_t width_radius = width >> 1;
    uint8_t height_radius = height >> 1;

    // Find how many offsets from the pre-computed pattern we need
    size_t count = 0;
    for (size_t i = 0; i < pattern.size; ++i) {
      const auto& offset = pattern.offsets[i];
      if (std::abs(offset.r_offset) <= height_radius && std::abs(offset.c_offset) <= width_radius) {
        count++;
      } else {
        // Since offsets are sorted by distance, once we're outside bounds,
        // all remaining offsets will also be outside
        break;
      }
    }

    begin_ptr = pattern.offsets.data();
    end_ptr = begin_ptr + count;
  }

  // Iterator interface
  const ObservationPattern::Offset* begin() const {
    return begin_ptr;
  }
  const ObservationPattern::Offset* end() const {
    return end_ptr;
  }

  // Also support span-like interface
  size_t size() const {
    return end_ptr - begin_ptr;
  }
  const ObservationPattern::Offset& operator[](size_t idx) const {
    return begin_ptr[idx];
  }
};

}  // namespace PackedCoordinate
#endif  // PACKED_COORDINATE_HPP_
