#include <gtest/gtest.h>

#include "core/grid_object.hpp"

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
  std::vector<int> tags;  // Empty tags vector
  obj.init(1, "object", loc, tags);

  EXPECT_EQ(1, obj.type_id);
  EXPECT_EQ("object", obj.type_name);
  EXPECT_EQ(5, obj.location.r);
  EXPECT_EQ(10, obj.location.c);
  EXPECT_EQ(2, obj.location.layer);
}
