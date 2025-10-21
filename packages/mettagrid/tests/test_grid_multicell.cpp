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
};

// Test that arbitrarily many connected cells can be occupied.
TEST_F(GridMultiCellTest, OccupyUnlimitedConnectedCells) {
  TestObject* obj = new TestObject();
  obj->location = GridLocation(5, 5, 0);
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
    EXPECT_TRUE(grid->occupy_cell(*obj, loc));
  }
  EXPECT_EQ(path.size(), obj->extra_cells.size());
}

// Disallow diagonal-only connectivity when adding cells.
TEST_F(GridMultiCellTest, OccupyRejectsDisconnectedCell) {
  TestObject* obj = new TestObject();
  obj->location = GridLocation(5, 5, 0);
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);

  // First add a directly adjacent cell.
  EXPECT_TRUE(grid->occupy_cell(*obj, GridLocation(5, 6, 0)));

  // Attempt to add a cell that touches only diagonally; should be rejected.
  EXPECT_FALSE(grid->occupy_cell(*obj, GridLocation(6, 7, 0)));
  EXPECT_EQ(1, obj->extra_cells.size());
}

// Removing a bridge cell should fail, while removing a leaf cell succeeds.
TEST_F(GridMultiCellTest, ReleaseCellPreservesConnectivity) {
  TestObject* obj = new TestObject();
  obj->location = GridLocation(5, 5, 0);
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);

  EXPECT_TRUE(grid->occupy_cell(*obj, GridLocation(5, 6, 0)));
  EXPECT_TRUE(grid->occupy_cell(*obj, GridLocation(5, 7, 0)));

  // Removing the middle cell would disconnect the tail; expect failure.
  EXPECT_FALSE(grid->release_cell(*obj, GridLocation(5, 6, 0)));
  EXPECT_EQ(2, obj->extra_cells.size());

  // Removing the outer cell is allowed.
  EXPECT_TRUE(grid->release_cell(*obj, GridLocation(5, 7, 0)));
  EXPECT_EQ(1, obj->extra_cells.size());
}

// Moving should fail for multi-cell objects because extra cells would be left behind.
TEST_F(GridMultiCellTest, MoveObjectRejectsMultiCell) {
  TestObject* obj = new TestObject();
  obj->location = GridLocation(3, 3, 0);
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->occupy_cell(*obj, GridLocation(3, 4, 0)));

  GridLocation destination = GridLocation(4, 3, 0);
  EXPECT_TRUE(grid->is_valid_location(destination));
  EXPECT_TRUE(grid->is_empty(destination.r, destination.c));

  EXPECT_FALSE(grid->move_object(*obj, destination));

  // Object and grid occupancy remain unchanged.
  EXPECT_EQ(GridLocation(3, 3, 0), obj->location);
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 3, 0)));
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 4, 0)));
}

// Swapping must fail when either object occupies multiple cells.
TEST_F(GridMultiCellTest, SwapObjectsRejectsMultiCellParticipants) {
  // Multi-cell actor with anchor and one extra cell.
  TestObject* multi = new TestObject();
  multi->location = GridLocation(2, 2, 0);
  multi->type_id = 1;
  multi->type_name = "multi";

  ASSERT_TRUE(grid->add_object(multi));
  ASSERT_TRUE(grid->occupy_cell(*multi, GridLocation(2, 3, 0)));
  ASSERT_FALSE(multi->extra_cells.empty());

  // Single-cell target we attempt to swap with.
  TestObject* solo = new TestObject();
  solo->location = GridLocation(2, 4, 0);
  solo->type_id = 2;
  solo->type_name = "solo";

  ASSERT_TRUE(grid->add_object(solo));

  // Swap should fail because multi occupies more than one cell.
  EXPECT_FALSE(grid->swap_objects(*multi, *solo));

  // State of objects and grid must remain unchanged.
  EXPECT_EQ(GridLocation(2, 2, 0), multi->location);
  EXPECT_EQ(GridLocation(2, 4, 0), solo->location);
  EXPECT_EQ(1u, multi->extra_cells.size());
  EXPECT_EQ(multi, grid->object_at(GridLocation(2, 2, 0)));
  EXPECT_EQ(multi, grid->object_at(GridLocation(2, 3, 0)));
  EXPECT_EQ(solo, grid->object_at(GridLocation(2, 4, 0)));
}
