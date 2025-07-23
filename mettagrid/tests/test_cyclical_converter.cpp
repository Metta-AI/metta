#include <gtest/gtest.h>

#include "event.hpp"
#include "grid.hpp"
#include "objects/converter.hpp"
#include "objects/production_handler.hpp"
#include "objects/constants.hpp"

namespace {

// Test resource IDs
const InventoryItem HEART = 1;
const InventoryItem BATTERY = 2;
const InventoryItem ORE = 3;

class CyclicalConverterTest : public ::testing::Test {
protected:
  void SetUp() override {
    grid = std::make_unique<Grid>(10, 10);
    event_manager = std::make_unique<EventManager>();
    event_manager->init(grid.get());

    // Set up event handlers
    event_manager->event_handlers.insert(
        {EventType::FinishConverting, std::make_unique<ProductionHandler>(event_manager.get())});
    event_manager->event_handlers.insert(
        {EventType::CoolDown, std::make_unique<CoolDownHandler>(event_manager.get())});

    current_time = 0;
  }

  std::unique_ptr<Grid> grid;
  std::unique_ptr<EventManager> event_manager;
  unsigned int current_time;
};

TEST_F(CyclicalConverterTest, BasicCyclicalBehavior) {
  // Create a cyclical converter (type_id = 100)
  ConverterConfig cfg(100, "cyclical_heart", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Initial state: should start converting
  EXPECT_TRUE(converter->converting);
  EXPECT_FALSE(converter->cooling_down);
  EXPECT_TRUE(converter->inventory.empty());

  // Step 1-4: Still converting
  for (int i = 1; i <= 4; i++) {
    current_time = i;
    event_manager->process_events(current_time);
    EXPECT_TRUE(converter->converting);
    EXPECT_FALSE(converter->cooling_down);
    EXPECT_TRUE(converter->inventory.empty());
  }

  // Step 5: Conversion completes, heart produced, cooldown starts
  current_time = 5;
  event_manager->process_events(current_time);
  EXPECT_FALSE(converter->converting);
  EXPECT_TRUE(converter->cooling_down);
  EXPECT_EQ(converter->inventory[HEART], 1);

  // Step 6-14: In cooldown, heart still available
  for (int i = 6; i <= 14; i++) {
    current_time = i;
    event_manager->process_events(current_time);
    EXPECT_FALSE(converter->converting);
    EXPECT_TRUE(converter->cooling_down);
    EXPECT_EQ(converter->inventory[HEART], 1);
  }

  // Step 15: Cooldown completes, inventory auto-emptied, starts converting again
  current_time = 15;
  event_manager->process_events(current_time);
  EXPECT_TRUE(converter->converting);
  EXPECT_FALSE(converter->cooling_down);
  EXPECT_TRUE(converter->inventory.empty());  // Auto-emptied!

  // Step 20: Second cycle completes
  current_time = 20;
  event_manager->process_events(current_time);
  EXPECT_FALSE(converter->converting);
  EXPECT_TRUE(converter->cooling_down);
  EXPECT_EQ(converter->inventory[HEART], 1);
}

TEST_F(CyclicalConverterTest, NonCyclicalConverterUnaffected) {
  // Create a normal converter (type_id = 5)
  ConverterConfig cfg(5, "normal_converter", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Let it complete a full cycle
  current_time = 5;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);

  // Complete cooldown - inventory should NOT be emptied
  current_time = 15;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);  // Still there!
  EXPECT_FALSE(converter->converting);  // Blocked by max_output
}

TEST_F(CyclicalConverterTest, CyclicalWithInputResources) {
  // Create a cyclical converter that requires ore input
  ConverterConfig cfg(102, "cyclical_ore_processor", {{ORE, 3}}, {{HEART, 1}}, 1, 5, 10, 0, 11);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Should not start converting without input
  EXPECT_FALSE(converter->converting);
  EXPECT_FALSE(converter->cooling_down);

  // Add ore
  converter->update_inventory(ORE, 3);

  // Should start converting
  EXPECT_TRUE(converter->converting);
  EXPECT_EQ(converter->inventory[ORE], 0);  // Consumed

  // Complete conversion
  current_time = 5;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);
  EXPECT_TRUE(converter->cooling_down);

  // Complete cooldown - both heart and any remaining resources cleared
  current_time = 15;
  event_manager->process_events(current_time);
  EXPECT_TRUE(converter->inventory.empty());  // Everything cleared

  // Won't start again without more ore
  EXPECT_FALSE(converter->converting);
}

TEST_F(CyclicalConverterTest, MultipleOutputsCyclical) {
  // Create a cyclical converter with multiple outputs
  ConverterConfig cfg(103, "cyclical_dual", {}, {{HEART, 1}, {BATTERY, 2}}, 3, 5, 10, 0, 14);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Complete conversion
  current_time = 5;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);
  EXPECT_EQ(converter->inventory[BATTERY], 2);

  // Complete cooldown - all outputs cleared
  current_time = 15;
  event_manager->process_events(current_time);
  EXPECT_TRUE(converter->inventory.empty());
  EXPECT_TRUE(converter->converting);  // Started again
}

TEST_F(CyclicalConverterTest, StatsTracking) {
  // Create a cyclical converter for stats testing
  ConverterConfig cfg(100, "cyclical_stats", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Complete two full cycles
  current_time = 5;
  event_manager->process_events(current_time);
  current_time = 15;
  event_manager->process_events(current_time);
  current_time = 20;
  event_manager->process_events(current_time);
  current_time = 30;
  event_manager->process_events(current_time);

  // Check stats
  auto stats = converter->stats.to_dict();
  EXPECT_EQ(stats["conversions.started"], 3.0f);  // Started 3 times
  EXPECT_EQ(stats["conversions.completed"], 2.0f);  // Completed 2 times
  EXPECT_EQ(stats["cooldown.completed"], 2.0f);  // Cooldown completed 2 times
  EXPECT_EQ(stats["inventory.auto_emptied"], 2.0f);  // Auto-emptied 2 times
  EXPECT_EQ(stats["heart.produced"], 2.0f);  // Produced 2 hearts
}

TEST_F(CyclicalConverterTest, BoundaryTypeIds) {
  // Test type_id = 100 (minimum cyclical)
  ConverterConfig cfg1(100, "cyclical_min", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter1 = new Converter(1, 1, cfg1);
  grid->add_object(converter1);
  converter1->set_event_manager(event_manager.get());

  // Test type_id = 199 (maximum cyclical)
  ConverterConfig cfg2(199, "cyclical_max", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter2 = new Converter(2, 2, cfg2);
  grid->add_object(converter2);
  converter2->set_event_manager(event_manager.get());

  // Test type_id = 99 (not cyclical)
  ConverterConfig cfg3(99, "not_cyclical_99", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter3 = new Converter(3, 3, cfg3);
  grid->add_object(converter3);
  converter3->set_event_manager(event_manager.get());

  // Test type_id = 200 (not cyclical)
  ConverterConfig cfg4(200, "not_cyclical_200", {}, {{HEART, 1}}, 1, 5, 10, 0, 13);
  auto* converter4 = new Converter(4, 4, cfg4);
  grid->add_object(converter4);
  converter4->set_event_manager(event_manager.get());

  // Let all complete conversion and cooldown
  current_time = 5;
  event_manager->process_events(current_time);
  current_time = 15;
  event_manager->process_events(current_time);

  // Check which ones auto-emptied
  EXPECT_TRUE(converter1->inventory.empty());   // type_id 100 - cyclical
  EXPECT_TRUE(converter2->inventory.empty());   // type_id 199 - cyclical
  EXPECT_FALSE(converter3->inventory.empty());  // type_id 99 - not cyclical
  EXPECT_FALSE(converter4->inventory.empty());  // type_id 200 - not cyclical
}

TEST_F(CyclicalConverterTest, NegativeCooldownStillWorks) {
  // Cyclical converter with negative cooldown (should stay in cooldown forever after first conversion)
  ConverterConfig cfg(105, "cyclical_negative", {}, {{HEART, 1}}, 1, 5, -1, 0, 13);
  auto* converter = new Converter(5, 5, cfg);
  grid->add_object(converter);
  converter->set_event_manager(event_manager.get());

  // Complete conversion
  current_time = 5;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);
  EXPECT_TRUE(converter->cooling_down);

  // Even after a long time, still in cooldown with heart
  current_time = 1000;
  event_manager->process_events(current_time);
  EXPECT_EQ(converter->inventory[HEART], 1);  // Not auto-emptied because cooldown never completes
  EXPECT_TRUE(converter->cooling_down);
}

}  // namespace
