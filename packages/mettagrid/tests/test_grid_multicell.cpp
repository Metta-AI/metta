#include <gtest/gtest.h>

#include "core/grid.hpp"
#include "core/grid_object.hpp"

// Test fixture for Grid multi-cell operations
class GridMultiCellTest : public ::testing::Test {
protected:
  void SetUp() override {
    grid = new Grid(10, 10);
  }

  void TearDown() override {
    delete grid;
  }

  Grid* grid;
};

// Concrete implementation of GridObject for testing
class TestObject : public GridObject {
public:
  std::vector<PartialObservationToken> obs_features() const override {
    return {};
  }

  // Test objects support multi-cell for testing
  bool supports_multi_cell() const override {
    return true;
  }
};

// Test that arbitrarily many cells can be occupied.
TEST_F(GridMultiCellTest, OccupyUnlimitedCells) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);

  std::vector<GridLocation> path = {
      GridLocation(5, 6, 0), GridLocation(5, 7, 0), GridLocation(5, 8, 0), GridLocation(5, 9, 0),
      GridLocation(4, 9, 0), GridLocation(3, 9, 0), GridLocation(2, 9, 0), GridLocation(1, 9, 0),
      GridLocation(1, 8, 0), GridLocation(1, 7, 0), GridLocation(1, 6, 0), GridLocation(1, 5, 0),
      GridLocation(1, 4, 0), GridLocation(2, 4, 0), GridLocation(3, 4, 0), GridLocation(4, 4, 0),
      GridLocation(4, 5, 0), GridLocation(4, 6, 0), GridLocation(4, 7, 0), GridLocation(4, 8, 0)};

  for (const auto& loc : path) {
    EXPECT_TRUE(grid->occupy_location(*obj, loc));
  }
  // locations includes the initial location plus extra cells added via path
  EXPECT_EQ(path.size() + 1, obj->locations.size());
}

// Allow cells at any location, including diagonal-only neighbors.
TEST_F(GridMultiCellTest, OccupyAllowsDisconnectedCells) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);

  // Add a directly adjacent cell.
  EXPECT_TRUE(grid->occupy_location(*obj, GridLocation(5, 6, 0)));

  // Add a cell that touches only diagonally - now allowed.
  EXPECT_TRUE(grid->occupy_location(*obj, GridLocation(6, 7, 0)));
  EXPECT_EQ(3u, obj->locations.size());

  // Add a completely disconnected cell - also allowed.
  EXPECT_TRUE(grid->occupy_location(*obj, GridLocation(1, 1, 0)));
  EXPECT_EQ(4u, obj->locations.size());
}

// Any cell can be removed except the last one.
TEST_F(GridMultiCellTest, ReleaseCellAllowsAnyNonLastCell) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);

  EXPECT_TRUE(grid->occupy_location(*obj, GridLocation(5, 6, 0)));
  EXPECT_TRUE(grid->occupy_location(*obj, GridLocation(5, 7, 0)));

  // Removing any cell is now allowed.
  EXPECT_TRUE(grid->release_location(*obj, GridLocation(5, 6, 0)));
  EXPECT_EQ(2u, obj->locations.size());

  // Removing another cell is allowed.
  EXPECT_TRUE(grid->release_location(*obj, GridLocation(5, 7, 0)));
  EXPECT_EQ(1u, obj->locations.size());

  // Cannot remove the last cell.
  EXPECT_FALSE(grid->release_location(*obj, GridLocation(5, 5, 0)));
  EXPECT_EQ(1u, obj->locations.size());
}

// Moving should fail for multi-cell objects because extra cells would be left behind.
TEST_F(GridMultiCellTest, MoveObjectRejectsMultiCell) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(3, 3, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->occupy_location(*obj, GridLocation(3, 4, 0)));

  GridLocation destination = GridLocation(4, 3, 0);
  EXPECT_TRUE(grid->is_valid_location(destination));
  EXPECT_TRUE(grid->is_empty_at_layer(destination.r, destination.c, obj->locations[0].layer));

  EXPECT_FALSE(grid->move_object(*obj, destination));

  // Object and grid occupancy remain unchanged.
  EXPECT_EQ(GridLocation(3, 3, 0), obj->locations[0]);
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 3, 0)));
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 4, 0)));
}

// Swapping must fail when either object occupies multiple cells.
TEST_F(GridMultiCellTest, SwapObjectsRejectsMultiCellParticipants) {
  // Multi-cell actor with initial location and one extra cell.
  TestObject* multi = new TestObject();
  multi->locations.push_back(GridLocation(2, 2, 0));
  multi->type_id = 1;
  multi->type_name = "multi";

  ASSERT_TRUE(grid->add_object(multi));
  ASSERT_TRUE(grid->occupy_location(*multi, GridLocation(2, 3, 0)));
  ASSERT_GE(multi->locations.size(), 2u);

  // Single-cell target we attempt to swap with.
  TestObject* solo = new TestObject();
  solo->locations.push_back(GridLocation(2, 4, 0));
  solo->type_id = 2;
  solo->type_name = "solo";

  ASSERT_TRUE(grid->add_object(solo));

  // Swap should fail because multi occupies more than one cell.
  EXPECT_FALSE(grid->swap_objects(*multi, *solo));

  // State of objects and grid must remain unchanged.
  EXPECT_EQ(GridLocation(2, 2, 0), multi->locations[0]);
  EXPECT_EQ(GridLocation(2, 4, 0), solo->locations[0]);
  EXPECT_EQ(2u, multi->locations.size());
  EXPECT_EQ(multi, grid->object_at(GridLocation(2, 2, 0)));
  EXPECT_EQ(multi, grid->object_at(GridLocation(2, 3, 0)));
  EXPECT_EQ(solo, grid->object_at(GridLocation(2, 4, 0)));
}

// Test that occupy_location rejects attempts to add cells on different layers
TEST_F(GridMultiCellTest, OccupyRejectsCrossLayerAddition) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(3, 3, 0));  // Layer 0
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_EQ(1u, obj->locations.size());
  EXPECT_EQ(0, obj->locations[0].layer);

  // Try to add a cell on a different layer
  GridLocation different_layer(3, 4, 1);  // Layer 1, adjacent position
  bool added = grid->occupy_location(*obj, different_layer);

  // Should fail - multi-cell objects must stay on same layer
  EXPECT_FALSE(added);
  EXPECT_EQ(1u, obj->locations.size());
  EXPECT_EQ(0, obj->locations[0].layer);
}
