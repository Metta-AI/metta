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

// Create a derived class for testing
class TestableGridObject : public GridObject {
public:
  // Make the protected encode method public for testing
  using GridObject::encode;

  // Implement the pure virtual method
  void obs(ObsType* obs) const override {
    // Simple implementation for testing
    encode(obs, "test_feature", 1);
  }
};

// Test fixture for GridObject
class GridObjectTest : public ::testing::Test {
protected:
  TestableGridObject obj;

  void SetUp() override {
    // Setup code if needed before each test
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

// Test feature registration and observation encoding
TEST_F(GridObjectTest, ObsMethod) {
  // Get initial observation size
  size_t initial_size = GridObject::get_observation_size();

  // Create an observation array large enough
  std::vector<ObsType> observations(initial_size + 1, 0);

  // Call the observation method
  obj.obs(observations.data());

  // Get the updated observation size (should be at least 1 more than before if feature didn't exist)
  size_t updated_size = GridObject::get_observation_size();
  EXPECT_GE(updated_size, initial_size);

  // Get the feature names to find our test_feature
  auto feature_names = GridObject::get_feature_names();

  // Find the index of our test_feature
  int test_feature_index = -1;
  for (size_t i = 0; i < feature_names.size(); i++) {
    if (feature_names[i] == "test_feature") {
      test_feature_index = i;
      break;
    }
  }

  // Verify the feature was registered and the observation was set
  ASSERT_GE(test_feature_index, 0) << "test_feature not found in feature names";
  EXPECT_EQ(1, observations[test_feature_index]);
}