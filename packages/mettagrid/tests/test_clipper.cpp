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
    unclip_recipe = std::make_shared<Recipe>(std::map<InventoryItem, InventoryQuantity>{{TestItems::ORE, 1}},
                                             std::map<InventoryItem, InventoryQuantity>{{TestItems::BATTERY, 1}},
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

  Clipper clipper(*grid, unclip_recipe, length_scale, cutoff_distance, clip_rate);

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

  Clipper clipper(*grid, unclip_recipe, length_scale, cutoff_distance, clip_rate);

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

  Clipper clipper(*grid, unclip_recipe, length_scale, cutoff_distance, clip_rate);

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

  Clipper clipper(*grid, unclip_recipe, length_scale, cutoff_distance, clip_rate);

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

  Clipper clipper(*grid, unclip_recipe, length_scale, cutoff_distance, clip_rate);

  // Give a_high_weight artificial high weight
  clipper.assembler_infection_weight[a_high_weight] = 100.0f;
  clipper.assembler_infection_weight[a_low_weight] = 0.1f;

  // Pick many times and verify high-weight assembler is picked more often
  std::mt19937 rng(42);
  int high_weight_picked = 0;
  int iterations = 100;

  for (int i = 0; i < iterations; i++) {
    Assembler* picked = clipper.pick_assembler_to_clip(rng);
    if (picked == a_high_weight) {
      high_weight_picked++;
    }
  }

  // With such a large weight difference, high_weight should be picked almost always
  EXPECT_GT(high_weight_picked, 90);  // At least 90% of the time
}
