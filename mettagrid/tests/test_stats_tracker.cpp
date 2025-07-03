#include <gtest/gtest.h>

#include "../mettagrid/stats_tracker.hpp"

// Mock MettaGrid for testing
class MockMettaGrid {
public:
  unsigned int current_step = 0;
};

// Test fixture for StatsTracker
class StatsTrackerTest : public ::testing::Test {
protected:
  StatsTracker stats;

  MockMettaGrid mock_env;

  void SetUp() override {
    stats.reset();
    mock_env.current_step = 0;
  }
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

// Test min/max tracking
TEST_F(StatsTrackerTest, MinMaxTracking) {
  stats.set("temperature", 20.0f);
  stats.set("temperature", 15.0f);
  stats.set("temperature", 25.0f);
  stats.set("temperature", 18.0f);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(18.0f, result["temperature"]);      // Current value
  EXPECT_FLOAT_EQ(15.0f, result["temperature.min"]);  // Minimum
  EXPECT_FLOAT_EQ(25.0f, result["temperature.max"]);  // Maximum
}

// Test average calculation
TEST_F(StatsTrackerTest, AverageCalculation) {
  // For tests that need timing, we'll test without environment
  // since we can't easily mock MettaGrid
  stats.add("points", 10);
  stats.add("points", 20);
  stats.add("points", 30);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(60.0f, result["points"]);  // Total

  // Without environment, we don't get timing metadata
  EXPECT_EQ(0, result.count("points.updates"));
  EXPECT_EQ(0, result.count("points.avg"));
}

// Test time tracking without environment =
TEST_F(StatsTrackerTest, TimingWithoutEnvironment) {
  // We can't easily test timing with a mock environment since
  // get_current_step() expects a real MettaGrid pointer.
  // This test will verify expected behavior when no environment
  // timing is available.

  stats.incr("action.move");
  stats.incr("action.move");
  stats.incr("action.move");

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(3.0f, result["action.move"]);

  // Without environment, no timing data
  EXPECT_EQ(0, result.count("action.move.first_step"));
  EXPECT_EQ(0, result.count("action.move.last_step"));
  EXPECT_EQ(0, result.count("action.move.updates"));
  EXPECT_EQ(0, result.count("action.move.rate"));
}

// Test rate calculation
TEST_F(StatsTrackerTest, RateCalculation) {
  // Without environment, rate should be 0
  for (int i = 0; i < 10; i++) {
    stats.incr("event");
  }

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(10.0f, result["event"]);
  EXPECT_EQ(0, result.count("event.rate"));  // No rate without environment
}

// Test reset functionality
TEST_F(StatsTrackerTest, ResetClearsAllData) {
  // Add some data
  stats.incr("counter");
  stats.set("value", 100);
  stats.add("score", 50);

  // Reset
  stats.reset();

  // All data should be cleared
  auto result = stats.to_dict();
  EXPECT_TRUE(result.empty());
}

// Test without environment (no timing)
TEST_F(StatsTrackerTest, NoEnvironmentNoTiming) {
  // Don't set environment
  stats.incr("action");
  stats.incr("action");

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(2.0f, result["action"]);
  EXPECT_EQ(0, result.count("action.first_step"));  // No timing data
  EXPECT_EQ(0, result.count("action.last_step"));
  EXPECT_EQ(0, result.count("action.rate"));
}

// Test complex stat keys
TEST_F(StatsTrackerTest, ComplexStatKeys) {
  stats.incr("action.attack.agent.red_team.blue_team");
  stats.add("inventory.armor.gained", 5);
  stats.set("status.health.current", 85.5f);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(1.0f, result["action.attack.agent.red_team.blue_team"]);
  EXPECT_FLOAT_EQ(5.0f, result["inventory.armor.gained"]);
  EXPECT_FLOAT_EQ(85.5f, result["status.health.current"]);
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
  EXPECT_FLOAT_EQ(-10.0f, result["negative.min"]);
  EXPECT_FLOAT_EQ(-5.0f, result["negative.max"]);
}

// Test large numbers
TEST_F(StatsTrackerTest, LargeNumbers) {
  stats.add("large", 1000000);
  stats.add("large", 2000000);

  auto result = stats.to_dict();
  EXPECT_FLOAT_EQ(3000000.0f, result["large"]);

  // Without environment, no average calculation
  EXPECT_EQ(0, result.count("large.avg"));
  EXPECT_EQ(0, result.count("large.updates"));
}

// Test all metadata is generated correctly
TEST_F(StatsTrackerTest, CompleteMetadata) {
  // Without a real MettaGrid, we can only test non-timing metadata
  stats.add("resource", 10);
  stats.add("resource", 20);
  stats.add("resource", 15);
  stats.add("resource", 5);

  auto result = stats.to_dict();

  // Check basic values
  EXPECT_FLOAT_EQ(50.0f, result["resource"]);      // Total: 10+20+15+5
  EXPECT_FLOAT_EQ(10.0f, result["resource.min"]);  // Min cumulative value
  EXPECT_FLOAT_EQ(50.0f, result["resource.max"]);  // Max cumulative value

  // Timing-related metadata won't be present without environment
  EXPECT_EQ(0, result.count("resource.avg"));
  EXPECT_EQ(0, result.count("resource.first_step"));
  EXPECT_EQ(0, result.count("resource.last_step"));
  EXPECT_EQ(0, result.count("resource.updates"));
  EXPECT_EQ(0, result.count("resource.rate"));
}
