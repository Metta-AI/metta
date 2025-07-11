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

  // Pre-sorted offsets organized so that square windows can use a contiguous prefix
  std::array<Offset, MAX_PATTERN_SIZE> offsets;
  size_t size;  // Actual number of valid offsets

  // For each possible square window size (1x1, 3x3, 5x5, ..., 15x15),
  // store the end index in the offsets array
  // Window sizes: 1x1=1, 3x3=9, 5x5=25, 7x7=49, 9x9=81, 11x11=121, 13x13=169, 15x15=225
  std::array<size_t, 8> window_end_indices;  // For radius 0-7

  ObservationPattern() {
    size = 0;
    const int8_t max_radius = MAX_OBSERVABLE_SIZE / 2;  // 7 for 15x15

    // Build offsets organized by radius layers, then by distance within each layer
    struct OffsetInfo {
      int8_t r_offset;
      int8_t c_offset;
      uint8_t radius_needed;  // max(|r|, |c|) - determines which square window includes this
      uint8_t distance;       // |r| + |c| - for sorting within same radius
    };

    std::vector<OffsetInfo> temp_offsets;
    temp_offsets.reserve(MAX_PATTERN_SIZE);

    // Generate all positions
    for (int r = -max_radius; r <= max_radius; r++) {
      for (int c = -max_radius; c <= max_radius; c++) {
        uint8_t radius_needed = std::max(std::abs(r), std::abs(c));
        uint8_t distance = std::abs(r) + std::abs(c);
        temp_offsets.push_back({static_cast<int8_t>(r), static_cast<int8_t>(c), radius_needed, distance});
      }
    }

    // Sort by: 1) radius needed (which square window), 2) Manhattan distance within that radius
    std::stable_sort(temp_offsets.begin(), temp_offsets.end(), [](const OffsetInfo& a, const OffsetInfo& b) {
      if (a.radius_needed != b.radius_needed) {
        return a.radius_needed < b.radius_needed;
      }
      return a.distance < b.distance;
    });

    // Copy to our fixed array and record end indices for each radius
    std::fill(window_end_indices.begin(), window_end_indices.end(), 0);

    for (const auto& info : temp_offsets) {
      offsets[size++] = {info.r_offset, info.c_offset};
      // Update end index for this radius and all larger radii
      for (uint8_t r = info.radius_needed; r <= max_radius; ++r) {
        window_end_indices[r] = size;
      }
    }
  }
};

// Get the global pattern instance
inline const ObservationPattern& get_observation_pattern() {
  static const ObservationPattern pattern;
  return pattern;
}

/**
 * Lightweight view into the observation pattern for a specific square window size.
 *
 * This class provides zero-allocation iteration over observation offsets for square
 * windows. The allowed window sizes are 1x1, 3x3, 5x5, 7x7, 9x9, 11x11, 13x13, and 15x15.
 *
 * The offsets are pre-sorted so that each window size corresponds to a contiguous
 * prefix of the global pattern, enabling simple pointer-based iteration.
 */
class ObservationSearchPattern {
  static_assert(MAX_PACKABLE_COORD == 14, "This implementation assumes 15x15 max window");

private:
  const ObservationPattern::Offset* begin_ptr;
  const ObservationPattern::Offset* end_ptr;

public:
  explicit ObservationSearchPattern(uint8_t width) {
    // in debug, assert we have a valid odd width
    assert(width % 2 == 1 && "Window size must be odd");
    assert(width <= ObservationPattern::MAX_OBSERVABLE_SIZE && "Window size must be <= 15");

    const auto& pattern = get_observation_pattern();
    uint8_t radius = width >> 1;

    begin_ptr = pattern.offsets.data();
    end_ptr = begin_ptr + pattern.window_end_indices[radius];
  }

  // Simple contiguous iterator interface
  const ObservationPattern::Offset* begin() const {
    return begin_ptr;
  }
  const ObservationPattern::Offset* end() const {
    return end_ptr;
  }

  size_t size() const {
    return end_ptr - begin_ptr;
  }
  const ObservationPattern::Offset& operator[](size_t idx) const {
    return begin_ptr[idx];
  }
};

}  // namespace PackedCoordinate
#endif  // PACKED_COORDINATE_HPP_
