#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>

#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/protocol.hpp"
#include "systems/clipper.hpp"
#include "systems/stats_tracker.hpp"

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
    resource_names = {"ore", "battery"};
    stats_tracker = std::make_unique<StatsTracker>(&resource_names);
    unclip_protocol =
        std::make_shared<Protocol>(0,
                                   std::vector<ObservationType>{},
                                   std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::ORE, 1}},
                                   std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::BATTERY, 1}},
                                   10);
  }

  void TearDown() override {}

  std::unique_ptr<Grid> grid;
  std::shared_ptr<Protocol> unclip_protocol;
  unsigned int current_timestep;
  std::vector<std::string> resource_names;
  std::unique_ptr<StatsTracker> stats_tracker;

  // Helper to create an assembler at a specific location
  Assembler* create_assembler(GridCoord r, GridCoord c) {
    AssemblerConfig cfg(1, "test_assembler");
    // cfg.protocols is empty by default, which is fine for these tests

    Assembler* assembler = new Assembler(r, c, cfg, stats_tracker.get());
    grid->add_object(assembler);
    assembler->set_current_timestep_ptr(&current_timestep);
    return assembler;
  }
};

// Test: infection_weight calculation based on distance
TEST_F(ClipperTest, InfectionWeightCalculation) {
  GridCoord length_scale = 2u;
  uint32_t scaled_cutoff_distance = 2u;
  uint32_t clip_period = 10;

  // Create assemblers at different distances. Recall that we're using the L_inf norm.
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2_close = create_assembler(0, 1);   // Distance = 1 (zero length_scales)
  Assembler* a3_medium = create_assembler(3, 5);  // Distance = 5 (two length_scales)
  Assembler* a4_far = create_assembler(0, 7);     // Distance = 7 (three length_scales)

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_protocols, length_scale, scaled_cutoff_distance, clip_period, rng);

  // Test distance calculation
  GridCoord dist_close = clipper.distance(*a1, *a2_close);
  EXPECT_EQ(dist_close, 1u);

  GridCoord dist_medium = clipper.distance(*a1, *a3_medium);
  EXPECT_EQ(dist_medium, 5u);

  GridCoord dist_far = clipper.distance(*a1, *a4_far);
  EXPECT_EQ(dist_far, 7u);

  // Test infection weight calculation
  uint32_t weight_close = clipper.infection_weight(*a1, *a2_close);
  EXPECT_EQ(weight_close, 4u);  // 2^2

  uint32_t weight_medium = clipper.infection_weight(*a1, *a3_medium);
  EXPECT_EQ(weight_medium, 1u);  // 2^0

  // Beyond cutoff should be 0
  uint32_t weight_far = clipper.infection_weight(*a1, *a4_far);
  EXPECT_EQ(weight_far, 0u);
}

// Test: infection weights are properly adjusted during clipping
TEST_F(ClipperTest, WeightAdjustmentDuringClipping) {
  GridCoord length_scale = 2u;
  uint32_t scaled_cutoff_distance = 5u;
  uint32_t clip_period = 10;

  // Create three assemblers in a line
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);  // Distance 2 from a1
  Assembler* a3 = create_assembler(0, 4);  // Distance 4 from a1, distance 2 from a2

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_protocols, length_scale, scaled_cutoff_distance, clip_period, rng);

  // Initially all assemblers should have weight 0
  EXPECT_EQ(clipper.assembler_infection_weight[a1], 0);
  EXPECT_EQ(clipper.assembler_infection_weight[a2], 0);
  EXPECT_EQ(clipper.assembler_infection_weight[a3], 0);

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
  uint32_t expected_weight_a1 = clipper.infection_weight(*a2, *a1);
  uint32_t expected_weight_a3 = clipper.infection_weight(*a2, *a3);

  EXPECT_EQ(clipper.assembler_infection_weight[a1], expected_weight_a1);
  EXPECT_EQ(clipper.assembler_infection_weight[a3], expected_weight_a3);

  // a2's weight should remain 0 (it's clipped)
  EXPECT_EQ(clipper.assembler_infection_weight[a2], 0);
}

// Test: infection weights are properly adjusted during unclipping
TEST_F(ClipperTest, WeightAdjustmentDuringUnclipping) {
  GridCoord length_scale = 2u;
  uint32_t scaled_cutoff_distance = 5u;
  uint32_t clip_period = 10;

  // Create three assemblers
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);  // Distance 2 from a1
  Assembler* a3 = create_assembler(0, 4);  // Distance 4 from a1, distance 2 from a2

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_protocols, length_scale, scaled_cutoff_distance, clip_period, rng);

  // Clip a2
  clipper.clip_assembler(*a2);

  uint32_t weight_a1_after_clip = clipper.assembler_infection_weight[a1];
  uint32_t weight_a3_after_clip = clipper.assembler_infection_weight[a3];

  EXPECT_GT(weight_a1_after_clip, 0);
  EXPECT_GT(weight_a3_after_clip, 0);

  // Now unclip a2
  clipper.on_unclip_assembler(*a2);

  // a2 should be back in unclipped set
  EXPECT_EQ(clipper.unclipped_assemblers.count(a2), 1);

  // Weights of a1 and a3 should have decreased back to 0
  EXPECT_EQ(clipper.assembler_infection_weight[a1], 0);
  EXPECT_EQ(clipper.assembler_infection_weight[a3], 0);

  // a2's weight should still be 0
  EXPECT_EQ(clipper.assembler_infection_weight[a2], 0);
}

// Test: multiple clipped assemblers accumulate infection weights correctly
TEST_F(ClipperTest, MultipleClippedAssemblersAccumulateWeights) {
  GridCoord length_scale = 2u;
  uint32_t scaled_cutoff_distance = 5u;
  uint32_t clip_period = 10u;

  // Create four assemblers in a square
  Assembler* a1 = create_assembler(0, 0);
  Assembler* a2 = create_assembler(0, 2);
  Assembler* a3 = create_assembler(2, 0);
  Assembler* a4 = create_assembler(2, 2);  // Center-ish target

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_protocols, length_scale, scaled_cutoff_distance, clip_period, rng);

  // Clip a1 and a2
  clipper.clip_assembler(*a1);
  clipper.clip_assembler(*a2);

  // a4 should have accumulated weight from both a1 and a2
  uint32_t weight_from_a1 = clipper.infection_weight(*a1, *a4);
  uint32_t weight_from_a2 = clipper.infection_weight(*a2, *a4);
  uint32_t expected_total_weight = weight_from_a1 + weight_from_a2;

  EXPECT_EQ(clipper.assembler_infection_weight[a4], expected_total_weight);

  // a3 should only have weight from a1 (a2 is further away)
  uint32_t weight_from_a1_to_a3 = clipper.infection_weight(*a1, *a3);
  uint32_t weight_from_a2_to_a3 = clipper.infection_weight(*a2, *a3);

  EXPECT_EQ(clipper.assembler_infection_weight[a3], weight_from_a1_to_a3 + weight_from_a2_to_a3);
}

// Test: pick_assembler_to_clip respects weights
TEST_F(ClipperTest, PickAssemblerRespectsWeights) {
  GridCoord length_scale = 2u;
  uint32_t scaled_cutoff_distance = 5u;
  uint32_t clip_period = 1;  // Always clip

  // Create assemblers with one having much higher weight
  Assembler* a_high_weight = create_assembler(0, 0);
  Assembler* a_low_weight = create_assembler(5, 5);

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  std::mt19937 rng(42);
  Clipper clipper(*grid, unclip_protocols, length_scale, scaled_cutoff_distance, clip_period, rng);

  // Give a_high_weight artificial high weight
  clipper.assembler_infection_weight[a_high_weight] = 1000;
  clipper.border_assemblers.insert(a_high_weight);
  clipper.assembler_infection_weight[a_low_weight] = 1;
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
    resource_names = {"ore", "battery"};
    stats_tracker = std::make_unique<StatsTracker>(&resource_names);
    unclip_protocol =
        std::make_shared<Protocol>(0,
                                   std::vector<ObservationType>{},
                                   std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::ORE, 1}},
                                   std::unordered_map<InventoryItem, InventoryQuantity>{{TestItems::BATTERY, 1}},
                                   10);
  }

  void TearDown() override {}

  std::shared_ptr<Protocol> unclip_protocol;
  unsigned int current_timestep;
  std::mt19937 rng;
  std::vector<std::string> resource_names;
  std::unique_ptr<StatsTracker> stats_tracker;

  // Helper to create an assembler at a specific location
  Assembler* create_assembler(Grid& grid, GridCoord r, GridCoord c, bool clip_immune = false) {
    AssemblerConfig cfg(1, "test_assembler");
    // cfg.protocols is empty by default, which is fine for these tests
    cfg.clip_immune = clip_immune;

    Assembler* assembler = new Assembler(r, c, cfg, stats_tracker.get());
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

  Clipper clipper(grid, {unclip_protocol}, 0u, 0u, 10, rng);

  // Expected: 50 / 5 / 2 = 5
  EXPECT_EQ(clipper.length_scale, 5);
}

// Test 2: Manual length_scale (positive value)
TEST_F(ClipperPercolationTest, ManualLengthScale) {
  Grid grid(50, 50);
  place_assemblers(grid, 25);

  Clipper clipper(grid, {unclip_protocol}, 3u, 0u, 10, rng);

  EXPECT_EQ(clipper.length_scale, 3);
}

// Test 3: Clip-immune assemblers excluded from count
TEST_F(ClipperPercolationTest, ClipImmuneExcluded) {
  Grid grid(50, 50);

  // Place 20 normal assemblers
  place_assemblers(grid, 9, false);

  // Place 5 clip-immune assemblers
  place_assemblers(grid, 16, true);

  Clipper clipper(grid, {unclip_protocol}, 0u, 0u, 10, rng);

  // Should use 9 (not 25) for calculation
  // Expected: 50 / 3 / 2 = 8 (with rounding)
  EXPECT_EQ(clipper.length_scale, 8);
}

// Test 4: Grid size scaling (larger grid → larger length_scale)
TEST_F(ClipperPercolationTest, GridSizeScaling) {
  Grid grid_small(25, 25);
  Grid grid_large(100, 100);

  // Place 25 assemblers in each
  place_assemblers(grid_small, 25);
  place_assemblers(grid_large, 25);

  Clipper clipper_small(grid_small, {unclip_protocol}, 0u, 0u, 10, rng);
  Clipper clipper_large(grid_large, {unclip_protocol}, 0u, 0u, 10, rng);

  // Ratio should be 100/25 = 4x
  // Note that we use division here instead of multiplication to brush rounding errors under the rug.
  EXPECT_EQ(clipper_large.length_scale / 4, clipper_small.length_scale);
}
