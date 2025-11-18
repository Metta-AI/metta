#include <gtest/gtest.h>

#include "core/grid.hpp"
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
  TestGridObject(TypeId type_id,
                 const std::string& type_name,
                 const std::vector<GridLocation>& locs,
                 const std::vector<int>& tags)
      : GridObject(type_id, type_name, locs, tags) {}

  std::vector<PartialObservationToken> obs_features() const override {
    std::vector<PartialObservationToken> features;
    features.push_back({0, 1});
    return features;
  }
};

// Test construction with locations seeds the footprint
TEST(GridObjectTest, ConstructorSeedsLocationAndFootprint) {
  GridLocation loc(5, 10);
  std::vector<int> tags;  // Empty tags vector

  std::vector<GridLocation> locations;
  locations.push_back(loc);

  // Use a heap-allocated object since Grid takes ownership of added objects.
  auto* heap_obj = new TestGridObject(1, "object", locations, tags);

  EXPECT_EQ(1, heap_obj->type_id);
  EXPECT_EQ("object", heap_obj->type_name);

  // Object should have a seeded footprint before being added to the grid.
  ASSERT_EQ(heap_obj->locations.size(), 1u);
  EXPECT_EQ(5, heap_obj->locations[0].r);
  EXPECT_EQ(10, heap_obj->locations[0].c);

  Grid grid(20, 20);
  ASSERT_TRUE(grid.add_object(heap_obj));
  ASSERT_EQ(heap_obj->locations.size(), 1u);
  EXPECT_EQ(5, heap_obj->locations[0].r);
  EXPECT_EQ(10, heap_obj->locations[0].c);
}

TEST(GridObjectTest, ConstructorRejectsEmptyLocations) {
  std::vector<GridLocation> locations;
  std::vector<int> tags;

  EXPECT_DEATH({ TestGridObject obj(1, "object", locations, tags); }, ".*");
}
