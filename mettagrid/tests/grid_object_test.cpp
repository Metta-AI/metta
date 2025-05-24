#include "grid_object.hpp"

#include <gtest/gtest.h>

// Test fixture for GridLocation class
class GridLocationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Setup code if needed before each test
  }

  void TearDown() override {
    // Cleanup code if needed after each test
  }
};

// Test default constructor
TEST_F(GridLocationTest, DefaultConstructor) {
  GridLocation location;
  EXPECT_EQ(0, location.r);
  EXPECT_EQ(0, location.c);
}

// Test two-parameter constructor
TEST_F(GridLocationTest, TwoParamConstructor) {
  GridLocation location(5, 10);
  EXPECT_EQ(5, location.r);
  EXPECT_EQ(10, location.c);
}

// Concrete implementation of GridObject for testing
class TestGridObject : public GridObject {
public:
  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({1, 1});
    return features;
  }
  void obs(ObsType* obs, const vector<uint8_t>& offsets) const override {
    // Simple implementation for testing
    obs[offsets[0]] = 1;
  }
};

// Test fixture for GridObject
class GridObjectTest : public ::testing::Test {
protected:
  TestGridObject obj;

  void SetUp() override {
    // Reset object before each test if needed
  }
};

// Test init with GridLocation
TEST_F(GridObjectTest, InitWithLocation) {
  GridLocation loc(5, 10);
  obj.init(1, loc);

  EXPECT_EQ(1, obj._type_id);
  EXPECT_EQ(5, obj.location.r);
  EXPECT_EQ(10, obj.location.c);
}

// Test init with coordinates
TEST_F(GridObjectTest, InitWithCoordinates) {
  obj.init(2, 15, 20);

  EXPECT_EQ(2, obj._type_id);
  EXPECT_EQ(15, obj.location.r);
  EXPECT_EQ(20, obj.location.c);
}

// Test obs method
TEST_F(GridObjectTest, ObsMethod) {
  ObsType observations[1] = {0};
  vector<uint8_t> offsets = {0};

  obj.obs(observations, offsets);

  EXPECT_EQ(1, observations[0]);
}
