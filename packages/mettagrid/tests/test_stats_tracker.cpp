#include <gtest/gtest.h>

#include "systems/stats_tracker.hpp"

// Test fixture for StatsTracker
class StatsTrackerTest : public ::testing::Test {
protected:
  std::vector<std::string> resource_names = {"resource1", "resource2", "resource3"};
  StatsTracker stats{&resource_names};
};

// Test basic increment functionality
TEST_F(StatsTrackerTest, BasicIncrement) {
  stats.incr("test.counter");
  stats.incr("test.counter");
  stats.incr("test.counter");

  EXPECT_FLOAT_EQ(3.0f, stats.get("test.counter"));
}

// Test add with integers
TEST_F(StatsTrackerTest, AddIntegers) {
  stats.add("score", 10);
  stats.add("score", 15);
  stats.add("score", 25);

  EXPECT_FLOAT_EQ(50.0f, stats.get("score"));
}

// Test add with floats
TEST_F(StatsTrackerTest, AddFloats) {
  stats.add("damage", 10.5f);
  stats.add("damage", 15.3f);
  stats.add("damage", 24.2f);

  EXPECT_FLOAT_EQ(50.0f, stats.get("damage"));
}

// Test set operations
TEST_F(StatsTrackerTest, SetOperations) {
  stats.set("health", 100);
  stats.set("health", 85);
  stats.set("health", 90);

  EXPECT_FLOAT_EQ(90.0f, stats.get("health"));  // Should be the last set value
}

// Test edge cases
TEST_F(StatsTrackerTest, EdgeCases) {
  // Zero values
  stats.add("zero", 0);
  stats.add("zero", 0);

  // Negative values
  stats.add("negative", -10);
  stats.add("negative", 5);

  EXPECT_FLOAT_EQ(0.0f, stats.get("zero"));
  EXPECT_FLOAT_EQ(-5.0f, stats.get("negative"));
}

// Test large numbers
TEST_F(StatsTrackerTest, LargeNumbers) {
  stats.add("large", 1000000);
  stats.add("large", 2000000);

  EXPECT_FLOAT_EQ(3000000.0f, stats.get("large"));
}
