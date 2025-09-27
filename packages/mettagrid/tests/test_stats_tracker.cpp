#include <gtest/gtest.h>

#include "systems/stats_tracker.hpp"

// Test fixture for StatsTracker
class StatsTrackerTest : public ::testing::Test {
protected:
  StatsTracker stats;
};

// Test basic increment functionality
TEST_F(StatsTrackerTest, BasicIncrement) {
  stats.incr("test.counter");
  stats.incr("test.counter");
  stats.incr("test.counter");

  auto result = stats.to_dict();
  // Note: to_dict() returns all values as floats for Python compatibility,
  // so we use EXPECT_FLOAT_EQ even for integer stats
  EXPECT_FLOAT_EQ(3.0f, result["test.counter"]);
}

// Test add with integers
TEST_F(StatsTrackerTest, AddIntegers) {
  stats.add("score", 10);
  stats.add("score", 15);
  stats.add("score", 25);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(50.0f, result["score"]);
}

// Test add with floats
TEST_F(StatsTrackerTest, AddFloats) {
  stats.add("damage", 10.5f);
  stats.add("damage", 15.3f);
  stats.add("damage", 24.2f);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(50.0f, result["damage"]);
}

// Test set operations
TEST_F(StatsTrackerTest, SetOperations) {
  stats.set("health", 100);
  stats.set("health", 85);
  stats.set("health", 90);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(90.0f, result["health"]);  // Should be the last set value
}

// Test edge cases
TEST_F(StatsTrackerTest, EdgeCases) {
  // Zero values
  stats.add("zero", 0);
  stats.add("zero", 0);

  // Negative values
  stats.add("negative", -10);
  stats.add("negative", 5);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(0.0f, result["zero"]);
  EXPECT_FLOAT_EQ(-5.0f, result["negative"]);
}

// Test large numbers
TEST_F(StatsTrackerTest, LargeNumbers) {
  stats.add("large", 1000000);
  stats.add("large", 2000000);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(3000000.0f, result["large"]);
}
