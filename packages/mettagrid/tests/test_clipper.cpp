#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>

#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/recipe.hpp"
#include "systems/clipper.hpp"

// Test-specific inventory item types
namespace TestItems {
constexpr uint8_t ORE = 0;
constexpr uint8_t BATTERY = 1;
}  // namespace TestItems

class ClipperTest : public ::testing::Test {
protected:
  void SetUp() override {
    grid = std::make_unique<Grid>(10, 10);
    current_timestep = 0;
    unclip_recipe =
        std::make_shared<Recipe>(std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::ORE, 1}},
                                 std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::BATTERY, 1}},
                                 10);
  }

  void TearDown() override {}

  std::unique_ptr<Grid> grid;
  std::shared_ptr<Recipe> unclip_recipe;
  unsigned int current_timestep;

  // Helper to create an assembler at a specific location
  Assembler* create_assembler(GridCoord r, GridCoord c) {
    std::vector<std::shared_ptr<Recipe>> recipes(256, nullptr);
    AssemblerConfig cfg(1, "test_assembler");
    cfg.recipes = recipes;

    Assembler* assembler = new Assembler(r, c, cfg);
    grid->add_object(assembler);
    assembler->set_current_timestep_ptr(&current_timestep);
    return assembler;
  }
};

// Test: infection_weight calculation based on distance
TEST_F(ClipperTest, InfectionWeightCalculation) {
  float length_scale = 2.0f;
  float cutoff_distance = 5.0f;
  float clip_rate = 0.1f;

  // Create two assemblers at different distances
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2_close = create_assembler(0, 2);   // Distance = 2
  Assembler* a3_medium = create_assembler(3, 4);  // Distance = 5
  Assembler* a4_far = create_assembler(5, 5);     // Distance ~7.07

  std::vector<std::shared_ptr<Recipe>> unclip_recipes = {unclip_recipe};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_recipes, length_scale, cutoff_distance, clip_rate, rng);

  // Test distance calculation
  float dist_close = clipper.distance(*a1, *a2_close);
  EXPECT_FLOAT_EQ(dist_close, 2.0f);

  float dist_medium = clipper.distance(*a1, *a3_medium);
  EXPECT_FLOAT_EQ(dist_medium, 5.0f);

  float dist_far = clipper.distance(*a1, *a4_far);
  EXPECT_NEAR(dist_far, std::sqrt(50.0f), 0.001f);

  // Test infection weight calculation
  float weight_close = clipper.infection_weight(*a1, *a2_close);
  EXPECT_FLOAT_EQ(weight_close, std::exp(-2.0f / 2.0f));  // exp(-1)

  float weight_medium = clipper.infection_weight(*a1, *a3_medium);
  EXPECT_FLOAT_EQ(weight_medium, std::exp(-5.0f / 2.0f));  // exp(-2.5)

  // Beyond cutoff should be 0
  float weight_far = clipper.infection_weight(*a1, *a4_far);
  EXPECT_FLOAT_EQ(weight_far, 0.0f);
}

// Test: infection weights are properly adjusted during clipping
TEST_F(ClipperTest, WeightAdjustmentDuringClipping) {
  float length_scale = 2.0f;
  float cutoff_distance = 10.0f;
  float clip_rate = 0.1f;

  // Create three assemblers in a line
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);  // Distance 2 from a1
  Assembler* a3 = create_assembler(0, 4);  // Distance 4 from a1, distance 2 from a2

  std::vector<std::shared_ptr<Recipe>> unclip_recipes = {unclip_recipe};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_recipes, length_scale, cutoff_distance, clip_rate, rng);

  // Initially all assemblers should have weight 0
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a1], 0.0f);
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a2], 0.0f);
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a3], 0.0f);

  // All should be in unclipped set
  EXPECT_EQ(clipper.unclipped_assemblers.count(a1), 1);
  EXPECT_EQ(clipper.unclipped_assemblers.count(a2), 1);
  EXPECT_EQ(clipper.unclipped_assemblers.count(a3), 1);

  // Clip a2 (the middle one)
  clipper.clip_assembler(*a2);

  // a2 should now be clipped and removed from unclipped set
  EXPECT_TRUE(a2->is_clipped);
  EXPECT_EQ(clipper.unclipped_assemblers.count(a2), 0);

  // Other assemblers should have increased weights
  float expected_weight_a1 = clipper.infection_weight(*a2, *a1);  // exp(-2/2) = exp(-1)
  float expected_weight_a3 = clipper.infection_weight(*a2, *a3);  // exp(-2/2) = exp(-1)

  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a1], expected_weight_a1);
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a3], expected_weight_a3);

  // a2's weight should remain 0 (it's clipped)
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a2], 0.0f);
}

// Test: infection weights are properly adjusted during unclipping
TEST_F(ClipperTest, WeightAdjustmentDuringUnclipping) {
  float length_scale = 2.0f;
  float cutoff_distance = 10.0f;
  float clip_rate = 0.1f;

  // Create three assemblers
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);  // Distance 2 from a1
  Assembler* a3 = create_assembler(0, 4);  // Distance 4 from a1, distance 2 from a2

  std::vector<std::shared_ptr<Recipe>> unclip_recipes = {unclip_recipe};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_recipes, length_scale, cutoff_distance, clip_rate, rng);

  // Clip a2
  clipper.clip_assembler(*a2);

  float weight_a1_after_clip = clipper.assembler_infection_weight[a1];
  float weight_a3_after_clip = clipper.assembler_infection_weight[a3];

  EXPECT_GT(weight_a1_after_clip, 0.0f);
  EXPECT_GT(weight_a3_after_clip, 0.0f);

  // Now unclip a2
  clipper.on_unclip_assembler(*a2);

  // a2 should be back in unclipped set
  EXPECT_EQ(clipper.unclipped_assemblers.count(a2), 1);

  // Weights of a1 and a3 should have decreased back to 0
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a1], 0.0f);
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a3], 0.0f);

  // a2's weight should still be 0
  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a2], 0.0f);
}

// Test: multiple clipped assemblers accumulate infection weights correctly
TEST_F(ClipperTest, MultipleClippedAssemblersAccumulateWeights) {
  float length_scale = 2.0f;
  float cutoff_distance = 10.0f;
  float clip_rate = 0.1f;

  // Create four assemblers in a square
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);
  Assembler* a3 = create_assembler(2, 0);
  Assembler* a4 = create_assembler(2, 2);  // Center-ish target

  std::vector<std::shared_ptr<Recipe>> unclip_recipes = {unclip_recipe};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_recipes, length_scale, cutoff_distance, clip_rate, rng);

  // Clip a1 and a2
  clipper.clip_assembler(*a1);
  clipper.clip_assembler(*a2);

  // a4 should have accumulated weight from both a1 and a2
  float weight_from_a1 = clipper.infection_weight(*a1, *a4);
  float weight_from_a2 = clipper.infection_weight(*a2, *a4);
  float expected_total_weight = weight_from_a1 + weight_from_a2;

  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a4], expected_total_weight);

  // a3 should only have weight from a1 (a2 is further away)
  float weight_from_a1_to_a3 = clipper.infection_weight(*a1, *a3);
  float weight_from_a2_to_a3 = clipper.infection_weight(*a2, *a3);

  EXPECT_FLOAT_EQ(clipper.assembler_infection_weight[a3], weight_from_a1_to_a3 + weight_from_a2_to_a3);
}

// Test: pick_assembler_to_clip respects weights
TEST_F(ClipperTest, PickAssemblerRespectsWeights) {
  float length_scale = 2.0f;
  float cutoff_distance = 10.0f;
  float clip_rate = 1.0f;  // Always clip

  // Create assemblers with one having much higher weight
  Assembler* a_high_weight = create_assembler(0, 0);
  Assembler* a_low_weight = create_assembler(5, 5);

  std::vector<std::shared_ptr<Recipe>> unclip_recipes = {unclip_recipe};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_recipes, length_scale, cutoff_distance, clip_rate, rng);

  // Give a_high_weight artificial high weight
  clipper.assembler_infection_weight[a_high_weight] = 100.0f;
  clipper.border_assemblers.insert(a_high_weight);
  clipper.assembler_infection_weight[a_low_weight] = 0.1f;
  clipper.border_assemblers.insert(a_low_weight);

  // Pick many times and verify high-weight assembler is picked more often
  int high_weight_picked = 0;
  int iterations = 100;

  for (int i = 0; i < iterations; i++) {
    Assembler* picked = clipper.pick_assembler_to_clip();
    if (picked == a_high_weight) {
      high_weight_picked++;
    }
  }

  // With such a large weight difference, high_weight should be picked almost always
  EXPECT_GT(high_weight_picked, 90);  // At least 90% of the time
}

// ================================================================================================
// PERCOLATION-BASED LENGTH SCALE TESTS
// ================================================================================================

class ClipperPercolationTest : public ::testing::Test {
protected:
  void SetUp() override {
    current_timestep = 0;
    rng.seed(42);
    unclip_recipe =
        std::make_shared<Recipe>(std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::ORE, 1}},
                                 std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::BATTERY, 1}},
                                 10);
  }

  void TearDown() override {}

  std::shared_ptr<Recipe> unclip_recipe;
  unsigned int current_timestep;
  std::mt19937 rng;

  // Helper to create an assembler at a specific location
  Assembler* create_assembler(Grid& grid, GridCoord r, GridCoord c, bool clip_immune = false) {
    std::vector<std::shared_ptr<Recipe>> recipes(256, nullptr);
    AssemblerConfig cfg(1, "test_assembler");
    cfg.recipes = recipes;
    cfg.clip_immune = clip_immune;

    Assembler* assembler = new Assembler(r, c, cfg);
    grid.add_object(assembler);
    assembler->set_current_timestep_ptr(&current_timestep);
    return assembler;
  }

  // Helper to place N assemblers in a grid (evenly spaced)
  void place_assemblers(Grid& grid, size_t count, bool clip_immune = false) {
    GridCoord width = grid.width;
    GridCoord height = grid.height;

    // Simple grid placement - spread assemblers evenly
    size_t sqrt_count = static_cast<size_t>(std::sqrt(count)) + 1;
    size_t placed = 0;

    for (size_t i = 0; i < sqrt_count && placed < count; i++) {
      for (size_t j = 0; j < sqrt_count && placed < count; j++) {
        GridCoord r = (i + 1) * height / (sqrt_count + 1);
        GridCoord c = (j + 1) * width / (sqrt_count + 1);
        create_assembler(grid, r, c, clip_immune);
        placed++;
      }
    }
  }
};

// Test 1: Auto-calculation enabled with valid buildings
TEST_F(ClipperPercolationTest, AutoLengthScaleBasic) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Expected: (50 / sqrt(25)) * sqrt(4.51 / (4*π)) ≈ 5.991
  EXPECT_NEAR(clipper.length_scale, 5.991f, 0.01f);
}

// Test 2: Manual length_scale (positive value)
TEST_F(ClipperPercolationTest, ManualLengthScale) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_recipe}, 3.14f, 0.0f, 0.1f, rng);

  EXPECT_FLOAT_EQ(clipper.length_scale, 3.14f);
}

// Test 3: Clip-immune assemblers excluded from count
TEST_F(ClipperPercolationTest, ClipImmuneExcluded) {
  Grid grid(50, 50);

  // Place 20 normal assemblers
  place_assemblers(grid, 20, false);

  // Place 5 clip-immune assemblers
  place_assemblers(grid, 5, true);

  Clipper clipper(grid, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Should use 20 (not 25) for calculation
  // Expected: (50 / sqrt(20)) * sqrt(4.51 / (4*π)) ≈ 6.704
  EXPECT_NEAR(clipper.length_scale, 6.704f, 0.01f);
}

// Test 4: No assemblers - keeps provided value
TEST_F(ClipperPercolationTest, NoAssemblers) {
  Grid grid(50, 50);
  // Don't place any assemblers

  Clipper clipper(grid, {unclip_recipe}, 2.5f, 0.0f, 0.1f, rng);

  // Should keep provided length_scale when no buildings (no auto-calculation)
  EXPECT_FLOAT_EQ(clipper.length_scale, 2.5f);
}

// Test 5: Grid size scaling (larger grid → larger length_scale)
TEST_F(ClipperPercolationTest, GridSizeScaling) {
  Grid grid_small(25, 25);
  Grid grid_large(100, 100);

  // Place 25 assemblers in each
  place_assemblers(grid_small, 25);
  place_assemblers(grid_large, 25);

  Clipper clipper_small(grid_small, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);
  Clipper clipper_large(grid_large, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Ratio should be 100/25 = 4x
  float ratio = clipper_large.length_scale / clipper_small.length_scale;
  EXPECT_NEAR(ratio, 4.0f, 0.01f);
}

// Test 6: Building density scaling (more buildings → smaller length_scale)
TEST_F(ClipperPercolationTest, BuildingDensityScaling) {
  Grid grid_sparse(50, 50);
  Grid grid_dense(50, 50);

  // Place 25 vs 100 assemblers
  place_assemblers(grid_sparse, 25);
  place_assemblers(grid_dense, 100);

  Clipper clipper_sparse(grid_sparse, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);
  Clipper clipper_dense(grid_dense, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Ratio should be sqrt(100/25) = 2x (sparse should be larger)
  float ratio = clipper_sparse.length_scale / clipper_dense.length_scale;
  EXPECT_NEAR(ratio, 2.0f, 0.01f);
}

// Test 7: Non-square grids use max dimension
TEST_F(ClipperPercolationTest, NonSquareGrid) {
  Grid grid_horizontal(30, 60);  // width > height
  Grid grid_vertical(60, 30);    // height > width

  place_assemblers(grid_horizontal, 25);
  place_assemblers(grid_vertical, 25);

  Clipper clipper_horizontal(grid_horizontal, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);
  Clipper clipper_vertical(grid_vertical, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Both should use max(width, height) = 60, so should have same length_scale
  EXPECT_FLOAT_EQ(clipper_horizontal.length_scale, clipper_vertical.length_scale);

  // Expected: (60 / sqrt(25)) * sqrt(4.51 / (4*π)) ≈ 7.189
  EXPECT_NEAR(clipper_horizontal.length_scale, 7.189f, 0.01f);
}

// ================================================================================================
// CUTOFF DISTANCE AUTO-CALCULATION TESTS
// ================================================================================================

// Test 8: Auto-calculate cutoff_distance = 3 * length_scale
TEST_F(ClipperPercolationTest, AutoCutoffDistance) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_recipe}, -1.0f, 0.0f, 0.1f, rng);

  // Expected cutoff: 3 * 5.991 ≈ 17.973
  EXPECT_NEAR(clipper.cutoff_distance, 3.0f * clipper.length_scale, 0.001f);
  EXPECT_NEAR(clipper.cutoff_distance, 17.973f, 0.03f);
}

// Test 9: Manual cutoff_distance is preserved
TEST_F(ClipperPercolationTest, ManualCutoffDistance) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_recipe}, -1.0f, 20.0f, 0.1f, rng);

  // Should keep the provided cutoff_distance
  EXPECT_FLOAT_EQ(clipper.cutoff_distance, 20.0f);
}

// Test 10: Auto-cutoff with manual length_scale
TEST_F(ClipperPercolationTest, AutoCutoffWithManualLength) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_recipe}, 5.0f, 0.0f, 0.1f, rng);

  // Should auto-calculate cutoff as 3 * 5.0
  EXPECT_FLOAT_EQ(clipper.cutoff_distance, 15.0f);
}
