#include <gtest/gtest.h>

#include "../mettagrid/grid_object.hpp"

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
  EXPECT_EQ(0, location.layer);
}

// Test two-parameter constructor
TEST_F(GridLocationTest, TwoParamConstructor) {
  GridLocation location(5, 10);
  EXPECT_EQ(5, location.r);
  EXPECT_EQ(10, location.c);
  EXPECT_EQ(0, location.layer);  // Default layer should be 0
}

// Test three-parameter constructor
TEST_F(GridLocationTest, ThreeParamConstructor) {
  GridLocation location(5, 10, 2);
  EXPECT_EQ(5, location.r);
  EXPECT_EQ(10, location.c);
  EXPECT_EQ(2, location.layer);
}

// Concrete implementation of GridObject for testing
class TestGridObject : public GridObject {
public:
  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({0, 1});
    return features;
  }
  void obs(ObsType* obs) const override {
    // Simple implementation for testing
    obs[0] = 1;
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
  GridLocation loc(5, 10, 2);
  obj.init(1, loc);

  EXPECT_EQ(1, obj._type_id);
  EXPECT_EQ(5, obj.location.r);
  EXPECT_EQ(10, obj.location.c);
  EXPECT_EQ(2, obj.location.layer);
}

// Test init with coordinates
TEST_F(GridObjectTest, InitWithCoordinates) {
  obj.init(2, 15, 20);

  EXPECT_EQ(2, obj._type_id);
  EXPECT_EQ(15, obj.location.r);
  EXPECT_EQ(20, obj.location.c);
  EXPECT_EQ(0, obj.location.layer);  // Default layer
}

// Test init with coordinates and layer
TEST_F(GridObjectTest, InitWithCoordinatesAndLayer) {
  obj.init(3, 25, 30, 4);

  EXPECT_EQ(3, obj._type_id);
  EXPECT_EQ(25, obj.location.r);
  EXPECT_EQ(30, obj.location.c);
  EXPECT_EQ(4, obj.location.layer);
}

// Test obs method
TEST_F(GridObjectTest, ObsMethod) {
  ObsType observations[1] = {0};

  obj.obs(observations);

  EXPECT_EQ(1, observations[0]);
}
