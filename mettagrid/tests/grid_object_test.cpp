#include "grid_object.hpp"

#include <gtest/gtest.h>

#define TEST_FEATURE GridFeature::HP
#define TEST_FEATURE_NAME "hp"

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
  void obs(c_observations_type* obs) const override {
    encode(obs, GridFeature::HP, 1);  // Using HP as an example
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
  std::cout << "Initial observation size: " << initial_size << std::endl;

  // Create an observation array large enough
  std::vector<c_observations_type> observations(initial_size + 1, 0);

  // Call the observation method
  obj.obs(observations.data());

  // Get the updated observation size (should be at least 1 more than before if feature didn't exist)
  size_t updated_size = GridObject::get_observation_size();
  std::cout << "Updated observation size: " << updated_size << std::endl;

  // Get the feature names to find our test_feature
  auto feature_names = GridObject::get_feature_names();
  std::cout << "Feature names size: " << feature_names.size() << std::endl;

  // Print all feature names for debugging
  std::cout << "All feature names:" << std::endl;
  for (size_t i = 0; i < feature_names.size(); i++) {
    std::cout << "  [" << i << "]: " << feature_names[i] << std::endl;
  }

  // Find the index of our test_feature
  int test_feature_index = -1;
  for (size_t i = 0; i < feature_names.size(); i++) {
    if (feature_names[i] == TEST_FEATURE_NAME) {
      test_feature_index = i;
      break;
    }
  }

  // Print debug information about the test feature
  std::cout << "TEST_FEATURE_NAME: " << TEST_FEATURE_NAME << std::endl;
  std::cout << "test_feature_index: " << test_feature_index << std::endl;

  // If found, print the observation value at that index
  if (test_feature_index >= 0) {
    std::cout << "observations[" << test_feature_index << "]: " << static_cast<int>(observations[test_feature_index])
              << std::endl;
  }

  // Print the first few values of the observations array for debugging
  std::cout << "First few observation values:" << std::endl;
  for (size_t i = 0; i < std::min(initial_size + 1, static_cast<size_t>(10)); i++) {
    std::cout << "  [" << i << "]: " << static_cast<int>(observations[i]) << std::endl;
  }

  // Verify the feature was registered and the observation was set
  ASSERT_GE(test_feature_index, 0) << "test_feature not found in feature names";
  EXPECT_EQ(1, observations[test_feature_index]);
}

TEST_F(GridObjectTest, FeatureMapConsistency) {
  // Get all feature names
  const auto& feature_names = GridObject::get_feature_names();

  // Print size information for debugging
  std::cout << "GridFeature::COUNT: " << static_cast<int>(GridFeature::COUNT) << std::endl;
  std::cout << "GridFeatureNames.size(): " << feature_names.size() << std::endl;

  // First, check that the sizes match (COUNT should equal the size of GridFeatureNames)
  EXPECT_EQ(static_cast<size_t>(GridFeature::COUNT), feature_names.size())
      << "The COUNT value in GridFeature enum doesn't match the size of GridFeatureNames";

  // Now check specific mappings between enum values and feature names
  // Create a mapping of expected indices for important features
  std::unordered_map<std::string, int> expected_indices = {
      {"agent", static_cast<int>(GridFeature::AGENT)},
      {"agent:group", static_cast<int>(GridFeature::AGENT_GROUP)},
      {"hp", static_cast<int>(GridFeature::HP)},
      {"wall", static_cast<int>(GridFeature::WALL)},
      // Add more mappings as needed
  };

  // Check that each feature name is at the expected index
  for (const auto& [name, expected_index] : expected_indices) {
    int actual_index = -1;

    // Find the actual index of this feature name
    for (size_t i = 0; i < feature_names.size(); i++) {
      if (feature_names[i] == name) {
        actual_index = static_cast<int>(i);
        break;
      }
    }

    // Verify the indices match
    EXPECT_NE(-1, actual_index) << "Feature name '" << name << "' not found in GridFeatureNames";
    EXPECT_EQ(expected_index, actual_index)
        << "Feature '" << name << "' is at index " << actual_index << " but the enum value is " << expected_index;
  }
}