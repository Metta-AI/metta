#include <gtest/gtest.h>

#include "core/grid.hpp"
#include "core/grid_object.hpp"

// Test fixture for Grid presence management
class GridPresenceTest : public ::testing::Test {
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

TEST_F(GridPresenceTest, ObjectStartsPresent) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  bool added = grid->add_object(obj);
  EXPECT_TRUE(added);
  EXPECT_TRUE(obj->present_on_grid);
  EXPECT_EQ(obj, grid->object_at(GridLocation(5, 5, 0)));
}

TEST_F(GridPresenceTest, DeactivateObjectRemovesOccupancy) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(3, 3, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_EQ(obj, grid->object_at(GridLocation(3, 3, 0)));

  bool deactivated = grid->deactivate_object(*obj);
  EXPECT_TRUE(deactivated);
  EXPECT_FALSE(obj->present_on_grid);
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(3, 3, 0)));
}

TEST_F(GridPresenceTest, DeactivateMultiCellObjectClearsAll) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(4, 4, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->occupy_location(*obj, GridLocation(4, 5, 0)));
  ASSERT_TRUE(grid->occupy_location(*obj, GridLocation(5, 5, 0)));
  ASSERT_EQ(3u, obj->locations.size());

  bool deactivated = grid->deactivate_object(*obj);
  EXPECT_TRUE(deactivated);
  EXPECT_FALSE(obj->present_on_grid);
  EXPECT_TRUE(obj->locations.empty());
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(4, 4, 0)));
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(4, 5, 0)));
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(5, 5, 0)));
}

TEST_F(GridPresenceTest, ActivateObjectRestoresPreviousLocations) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(2, 2, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->deactivate_object(*obj));
  ASSERT_FALSE(obj->present_on_grid);

  bool activated = grid->activate_object(*obj);
  EXPECT_TRUE(activated);
  EXPECT_TRUE(obj->present_on_grid);
  EXPECT_EQ(obj, grid->object_at(GridLocation(2, 2, 0)));
}

TEST_F(GridPresenceTest, ActivateDoesNotRestoreExtraCells) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(6, 6, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->occupy_location(*obj, GridLocation(6, 7, 0)));
  ASSERT_EQ(2u, obj->locations.size());

  ASSERT_TRUE(grid->deactivate_object(*obj));
  ASSERT_TRUE(obj->locations.empty());

  bool activated = grid->activate_object(*obj);
  EXPECT_TRUE(activated);
  EXPECT_TRUE(obj->present_on_grid);
  // With previous_locations, full shape is restored
  ASSERT_EQ(2u, obj->locations.size());
  EXPECT_EQ(GridLocation(6, 6, 0), obj->locations[0]);
  EXPECT_EQ(GridLocation(6, 7, 0), obj->locations[1]);
  EXPECT_EQ(obj, grid->object_at(GridLocation(6, 6, 0)));
  EXPECT_EQ(obj, grid->object_at(GridLocation(6, 7, 0)));
}

TEST_F(GridPresenceTest, ActivateFailsIfPrimaryLocationOccupied) {
  TestObject* obj1 = new TestObject();
  obj1->locations.push_back(GridLocation(1, 1, 0));
  obj1->type_id = 1;
  obj1->type_name = "test1";

  TestObject* obj2 = new TestObject();
  obj2->locations.push_back(GridLocation(1, 1, 0));
  obj2->type_id = 2;
  obj2->type_name = "test2";

  ASSERT_TRUE(grid->add_object(obj1));
  ASSERT_TRUE(grid->deactivate_object(*obj1));

  // Add obj2 to the same location
  bool added = grid->add_object(obj2);
  EXPECT_TRUE(added);

  // Try to reactivate obj1 - should fail because primary location is occupied
  bool activated = grid->activate_object(*obj1);
  EXPECT_FALSE(activated);
  EXPECT_FALSE(obj1->present_on_grid);
  EXPECT_EQ(obj2, grid->object_at(GridLocation(1, 1, 0)));
}

TEST_F(GridPresenceTest, DeactivateTwiceIsIdempotent) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(7, 7, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));

  ASSERT_TRUE(grid->deactivate_object(*obj));
  EXPECT_FALSE(obj->present_on_grid);

  // Deactivate again should still succeed
  EXPECT_TRUE(grid->deactivate_object(*obj));
  EXPECT_FALSE(obj->present_on_grid);
}

TEST_F(GridPresenceTest, ActivateFailsIfLocationInvalid) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(100, 100, 0));  // Out of bounds
  obj->type_id = 1;
  obj->type_name = "test";
  obj->present_on_grid = false;

  // Don't add to grid, just try to activate with invalid location
  bool activated = grid->activate_object(*obj);
  EXPECT_FALSE(activated);
  EXPECT_FALSE(obj->present_on_grid);

  // Clean up since not added to grid
  delete obj;
}

// Test that add_object_location sets present_on_grid to true when successful
TEST_F(GridPresenceTest, AddObjectCellReactivatesDeactivatedObject) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(3, 3, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(obj->present_on_grid);

  // Deactivate the object
  ASSERT_TRUE(grid->deactivate_object(*obj));
  ASSERT_FALSE(obj->present_on_grid);
  ASSERT_EQ(nullptr, grid->object_at(GridLocation(3, 3, 0)));

  // Add a cell - should reactivate the object
  bool ok = grid->occupy_location(*obj, GridLocation(3, 4, 0));
  EXPECT_TRUE(ok);
  EXPECT_TRUE(obj->present_on_grid);
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 3, 0)));
  EXPECT_EQ(obj, grid->object_at(GridLocation(3, 4, 0)));
}

// Test that activate_object is idempotent
TEST_F(GridPresenceTest, ActivateTwiceIsIdempotent) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(obj->present_on_grid);

  // Activate again should still succeed
  EXPECT_TRUE(grid->activate_object(*obj));
  EXPECT_TRUE(obj->present_on_grid);
  EXPECT_EQ(obj, grid->object_at(GridLocation(5, 5, 0)));
}

// Test that operations fail on deactivated objects
TEST_F(GridPresenceTest, RemoveObjectCellFailsOnDeactivatedObject) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(2, 2, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->occupy_location(*obj, GridLocation(2, 3, 0)));
  ASSERT_EQ(2u, obj->locations.size());

  // Deactivate the object - this clears locations
  ASSERT_TRUE(grid->deactivate_object(*obj));
  ASSERT_TRUE(obj->locations.empty());

  // Try to remove a cell - should fail because no cells to remove
  bool removed = grid->release_location(*obj, GridLocation(2, 3, 0));
  EXPECT_FALSE(removed);
}

// Test that move_object fails on deactivated objects
TEST_F(GridPresenceTest, MoveObjectFailsOnDeactivatedObject) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(4, 4, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));
  ASSERT_TRUE(grid->deactivate_object(*obj));
  ASSERT_FALSE(obj->present_on_grid);

  // Try to move - should fail because object is not on grid
  GridLocation destination(5, 5, 0);
  bool moved = grid->move_object(*obj, destination);
  EXPECT_FALSE(moved);
  EXPECT_EQ(GridLocation(4, 4, 0), obj->previous_locations[0]);
  EXPECT_FALSE(obj->present_on_grid);
}

// Test that swap_objects fails when either object is deactivated
TEST_F(GridPresenceTest, SwapObjectsFailsWithDeactivatedObject) {
  TestObject* obj1 = new TestObject();
  obj1->locations.push_back(GridLocation(1, 1, 0));
  obj1->type_id = 1;
  obj1->type_name = "test1";

  TestObject* obj2 = new TestObject();
  obj2->locations.push_back(GridLocation(2, 2, 0));
  obj2->type_id = 2;
  obj2->type_name = "test2";

  ASSERT_TRUE(grid->add_object(obj1));
  ASSERT_TRUE(grid->add_object(obj2));

  // Deactivate obj1
  ASSERT_TRUE(grid->deactivate_object(*obj1));
  ASSERT_FALSE(obj1->present_on_grid);

  // Try to swap - should fail because obj1 is deactivated
  bool swapped = grid->swap_objects(*obj1, *obj2);
  EXPECT_FALSE(swapped);

  // Locations should remain unchanged
  EXPECT_EQ(GridLocation(1, 1, 0), obj1->previous_locations[0]);
  EXPECT_EQ(GridLocation(2, 2, 0), obj2->locations[0]);
  EXPECT_FALSE(obj1->present_on_grid);
  EXPECT_TRUE(obj2->present_on_grid);
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(1, 1, 0)));
  EXPECT_EQ(obj2, grid->object_at(GridLocation(2, 2, 0)));
}

// Test exhaustive occupancy verification for multi-cell objects
TEST_F(GridPresenceTest, ExhaustiveOccupancyVerification) {
  TestObject* obj = new TestObject();
  obj->locations.push_back(GridLocation(5, 5, 0));
  obj->type_id = 1;
  obj->type_name = "test";

  ASSERT_TRUE(grid->add_object(obj));

  // Build a 3x3 connected shape
  std::vector<GridLocation> cells_to_add = {GridLocation(5, 6, 0),
                                            GridLocation(5, 7, 0),
                                            GridLocation(6, 5, 0),
                                            GridLocation(6, 6, 0),
                                            GridLocation(6, 7, 0),
                                            GridLocation(7, 5, 0),
                                            GridLocation(7, 6, 0),
                                            GridLocation(7, 7, 0)};

  for (const auto& loc : cells_to_add) {
    ASSERT_TRUE(grid->occupy_location(*obj, loc));
  }

  // Verify all cells are occupied
  EXPECT_EQ(obj, grid->object_at(GridLocation(5, 5, 0)));
  for (const auto& loc : cells_to_add) {
    EXPECT_EQ(obj, grid->object_at(loc));
  }

  // Deactivate and verify ALL cells are cleared
  ASSERT_TRUE(grid->deactivate_object(*obj));
  EXPECT_FALSE(obj->present_on_grid);
  // locations cleared in deactivate

  // Exhaustively check that ALL cells are now nullptr
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(5, 5, 0)));
  for (const auto& loc : cells_to_add) {
    EXPECT_EQ(nullptr, grid->object_at(loc));
  }
}

// Test that when activate fails, present_on_grid stays false
TEST_F(GridPresenceTest, ActivateFailurePreservesState) {
  TestObject* obj1 = new TestObject();
  obj1->locations.push_back(GridLocation(8, 8, 0));
  obj1->type_id = 1;
  obj1->type_name = "test1";

  TestObject* obj2 = new TestObject();
  obj2->locations.push_back(GridLocation(8, 8, 0));
  obj2->type_id = 2;
  obj2->type_name = "test2";

  ASSERT_TRUE(grid->add_object(obj1));
  ASSERT_TRUE(grid->deactivate_object(*obj1));
  ASSERT_FALSE(obj1->present_on_grid);

  // Add obj2 to block the location
  ASSERT_TRUE(grid->add_object(obj2));

  // Try to activate obj1 - should fail
  bool activated = grid->activate_object(*obj1);
  EXPECT_FALSE(activated);

  // Verify state is consistent - obj1 still deactivated
  EXPECT_FALSE(obj1->present_on_grid);
  // The location should be occupied by obj2, not nullptr or obj1
  EXPECT_EQ(obj2, grid->object_at(GridLocation(8, 8, 0)));
}
