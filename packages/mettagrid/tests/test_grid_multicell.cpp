#include <gtest/gtest.h>

#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"

// Minimal multi-cell capable object used for grid footprint tests.
class MultiCellObject : public GridObject {
public:
  explicit MultiCellObject(std::vector<GridLocation> locations)
      : GridObject(/*type_id=*/1,
                   /*type_name=*/"multi",
                   std::move(locations),
                   /*tags=*/{},
                   /*object_vibe=*/0) {}

  bool supports_multi_cell() const override {
    return true;
  }
};

// Object type that does NOT opt into multi-cell footprints. Used to verify
// that Grid::add_object enforces supports_multi_cell() for multi-cell
// locations.
class SingleCellOnlyObject : public GridObject {
public:
  explicit SingleCellOnlyObject(std::vector<GridLocation> locations)
      : GridObject(/*type_id=*/2,
                   /*type_name=*/"single_only",
                   std::move(locations),
                   /*tags=*/{},
                   /*object_vibe=*/0) {}
};

class GridMulticellTest : public ::testing::Test {
protected:
  Grid grid{10, 10};
};

TEST_F(GridMulticellTest, AddObjectSeedsLocationsAndOccupiesCells) {
  std::vector<GridLocation> locations = {GridLocation(2, 3)};
  auto* obj = new MultiCellObject(locations);

  // Newly constructed object is not yet associated with a Grid. It should
  // already have a seeded single-cell footprint and an unset id.
  EXPECT_EQ(obj->id, 0);
  ASSERT_EQ(obj->locations.size(), 1u);
  EXPECT_EQ(obj->locations[0].r, 2);
  EXPECT_EQ(obj->locations[0].c, 3);

  // Adding to the grid assigns an id and marks the anchor cell as occupied.
  ASSERT_TRUE(grid.add_object(obj));
  EXPECT_NE(obj->id, 0);
  ASSERT_EQ(obj->locations.size(), 1u);
  EXPECT_EQ(obj->locations[0].r, 2);
  EXPECT_EQ(obj->locations[0].c, 3);
  EXPECT_EQ(grid.object(obj->id), obj);
  EXPECT_FALSE(grid.is_empty(2, 3));
}

TEST_F(GridMulticellTest, InitTimeMultiCellFootprintOccupiesAllCells) {
  // Configure a multi-cell footprint at construction time. This mirrors how
  // map builders or object factories would seed locations.
  std::vector<GridLocation> locations = {
      GridLocation(4, 4),
      GridLocation(4, 5),
      GridLocation(5, 4),
  };
  auto* obj = new MultiCellObject(locations);

  EXPECT_EQ(obj->id, 0);
  ASSERT_EQ(obj->locations.size(), 3u);

  ASSERT_TRUE(grid.add_object(obj));
  EXPECT_NE(obj->id, 0);
  ASSERT_EQ(obj->locations.size(), 3u);

  // Anchor location should remain the first entry.
  EXPECT_EQ(obj->locations[0].r, 4);
  EXPECT_EQ(obj->locations[0].c, 4);

  const GridLocation expected[] = {
      GridLocation(4, 4),
      GridLocation(4, 5),
      GridLocation(5, 4),
  };

  for (const auto& loc : expected) {
    EXPECT_FALSE(grid.is_empty(loc.r, loc.c));
    EXPECT_EQ(grid.object_at(loc), obj);
  }

  // A cell outside the configured footprint must remain empty.
  GridLocation outside(0, 0);
  EXPECT_TRUE(grid.is_empty(outside.r, outside.c));
}

TEST_F(GridMulticellTest, NonOptInTypesCannotBeMultiCell) {
  // Configure a multi-cell footprint for a type that has not opted in via
  // supports_multi_cell(). Grid::add_object should reject it.
  std::vector<GridLocation> locations = {
      GridLocation(1, 1),
      GridLocation(1, 2),
  };
  auto* obj = new SingleCellOnlyObject(locations);

  EXPECT_EQ(obj->id, 0);
  ASSERT_EQ(obj->locations.size(), 2u);
  EXPECT_FALSE(obj->supports_multi_cell());

  EXPECT_TRUE(grid.is_empty(1, 1));
  EXPECT_TRUE(grid.is_empty(1, 2));

  EXPECT_FALSE(grid.add_object(obj));
  EXPECT_EQ(obj->id, 0);  // Rejected objects must not be assigned an id.
  EXPECT_TRUE(grid.is_empty(1, 1));
  EXPECT_TRUE(grid.is_empty(1, 2));

  delete obj;
}
