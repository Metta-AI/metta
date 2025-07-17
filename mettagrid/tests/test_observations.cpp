#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "../mettagrid/packed_coordinate.hpp"  // Adjust path as needed

using PackedCoordinate::ObservationPattern;
using Offset = std::pair<int, int>;

int manhattan_distance(const Offset& offset) {
  return std::abs(offset.first) + std::abs(offset.second);
}

std::vector<std::pair<int, int>> compute_sorted_offsets(int height, int width) {
  const int row_min = -height / 2;
  const int row_max = height / 2;
  const int col_min = -width / 2;
  const int col_max = width / 2;

  std::vector<std::pair<int, int>> result;

  for (int dr = row_min; dr <= row_max; ++dr) {
    for (int dc = col_min; dc <= col_max; ++dc) {
      result.emplace_back(dr, dc);
    }
  }

  std::sort(result.begin(), result.end(), [](auto a, auto b) {
    int da = std::abs(a.first) + std::abs(a.second);
    int db = std::abs(b.first) + std::abs(b.second);
    if (da != db) return da < db;
    // Optional: stable tie-breaker
    return a < b;
  });

  return result;
}

TEST(ObservationPatternTest, OffsetsWithinBounds) {
  int height = 3;
  int width = 5;
  int row_min = -height / 2;
  int row_max = height / 2;
  int col_min = -width / 2;
  int col_max = width / 2;

  for (const auto& [dr, dc] : ObservationPattern{height, width}) {
    EXPECT_GE(dr, row_min);
    EXPECT_LE(dr, row_max);
    EXPECT_GE(dc, col_min);
    EXPECT_LE(dc, col_max);
  }
}

TEST(ObservationPatternTest, OffsetsAreUnique) {
  int height = 3;
  int width = 5;
  std::set<Offset> seen;

  for (const auto& offset : ObservationPattern{height, width}) {
    EXPECT_TRUE(seen.insert(offset).second) << "Duplicate offset: (" << offset.first << "," << offset.second << ")";
  }

  EXPECT_EQ(seen.size(), height * width);
}

TEST(ObservationPatternTest, OffsetsInManhattanOrder) {
  int height = 5;
  int width = 5;
  std::vector<int> distances;

  for (const auto& offset : ObservationPattern{height, width}) {
    distances.push_back(manhattan_distance(offset));
  }

  for (size_t i = 1; i < distances.size(); ++i) {
    EXPECT_LE(distances[i - 1], distances[i]) << "Offsets not in Manhattan order at index " << i;
  }
}

class ObservationPatternParamTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(ObservationPatternParamTest, MatchesReferenceOffsets) {
  int height = GetParam().first;
  int width = GetParam().second;

  auto expected = compute_sorted_offsets(height, width);
  std::vector<Offset> actual;
  for (const auto& offset : ObservationPattern{height, width}) {
    actual.push_back(offset);
  }

  EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_CASE_P(PackedCoordinate,
                        ObservationPatternParamTest,
                        ::testing::Values(std::make_pair(3, 9),
                                          std::make_pair(7, 3),
                                          std::make_pair(5, 5),
                                          std::make_pair(1, 1),
                                          std::make_pair(1, 5),
                                          std::make_pair(5, 1)),
                        [](const ::testing::TestParamInfo<std::pair<int, int>>& info) {
                          return "H" + std::to_string(info.param.first) + "_W" + std::to_string(info.param.second);
                        });
