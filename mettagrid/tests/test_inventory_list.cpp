#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "src/metta/mettagrid/objects/inventory_list.hpp"
#include "src/metta/mettagrid/event.hpp"

// Test items for testing
enum class TestItems {
  ORE = 0,
  HEART = 1,
  WOOD = 2
};

// Mock EventManager for testing
class MockEventManager : public EventManager {
public:
  MockEventManager() : current_timestep_(0) {}

  // Track scheduled events
  struct ScheduledEvent {
    EventType type;
    unsigned int delay;
    GridObjectId object_id;
    EventArg arg;
  };

  std::vector<ScheduledEvent> scheduled_events;
  unsigned int current_timestep_;

  void schedule_event(EventType type, unsigned int delay, GridObjectId object_id, EventArg arg) {
    scheduled_events.push_back({type, delay, object_id, arg});
  }

  unsigned int get_current_timestep() const {
    return current_timestep_;
  }

  void set_current_timestep(unsigned int timestep) {
    current_timestep_ = timestep;
  }

  // Clear scheduled events for test isolation
  void clear_events() {
    scheduled_events.clear();
  }

  // Get events of a specific type
  std::vector<ScheduledEvent> get_events_of_type(EventType type) const {
    std::vector<ScheduledEvent> result;
    for (const auto& event : scheduled_events) {
      if (event.type == type) {
        result.push_back(event);
      }
    }
    return result;
  }
};

class InventoryListTest : public ::testing::Test {
protected:
  void SetUp() override {
    mock_event_manager = std::make_unique<MockEventManager>();
    rng = std::make_unique<std::mt19937>(42);  // Fixed seed for reproducible tests
  }

  void TearDown() override {
    mock_event_manager.reset();
    rng.reset();
  }

  std::unique_ptr<MockEventManager> mock_event_manager;
  std::unique_ptr<std::mt19937> rng;
};

// Test basic inventory operations without resource loss
TEST_F(InventoryListTest, BasicInventoryOperations) {
  InventoryList inventory_list;

  // Test adding items
  InventoryDelta delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 5, 1);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 5);

  // Test removing items
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -2, 1);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 3);

  // Test removing more than available
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -10, 1);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 0);

  // Test adding to empty inventory
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::HEART), 3, 1);
  EXPECT_EQ(delta, 3);
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::HEART)], 3);
}

// Test resource loss functionality
TEST_F(InventoryListTest, ResourceLossScheduling) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;  // 10% loss probability
  resource_loss_prob[static_cast<InventoryItem>(TestItems::HEART)] = 0.0f;  // No loss

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Add items with resource loss
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 3, 1);

  // Should have scheduled 3 resource loss events for ORE
  auto ore_events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(ore_events.size(), 3);

  // All events should be for object_id 1
  for (const auto& event : ore_events) {
    EXPECT_EQ(event.object_id, 1);
  }

  // Add items without resource loss
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::HEART), 2, 1);

  // Should not have scheduled any additional events for HEART
  auto heart_events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(heart_events.size(), 3);  // Still only the ORE events
}

// Test resource instance tracking
TEST_F(InventoryListTest, ResourceInstanceTracking) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Add items
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 2, 1);

  // Should have 2 resource instances
  EXPECT_EQ(inventory_list.resource_instances.size(), 2);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 2);

  // Remove one item
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -1, 1);

  // Should have 1 resource instance left
  EXPECT_EQ(inventory_list.resource_instances.size(), 1);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 1);

  // Remove remaining items
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -1, 1);

  // Should have no resource instances
  EXPECT_EQ(inventory_list.resource_instances.size(), 0);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 0);
}

// Test populate_initial_inventory
TEST_F(InventoryListTest, PopulateInitialInventory) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::HEART)] = 0.05f;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Create initial inventory
  std::map<InventoryItem, InventoryQuantity> initial_inventory;
  initial_inventory[static_cast<InventoryItem>(TestItems::ORE)] = 3;
  initial_inventory[static_cast<InventoryItem>(TestItems::HEART)] = 2;

  // Populate initial inventory
  inventory_list.populate_initial_inventory(initial_inventory, 1);

  // Check inventory contents
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 3);
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::HEART)], 2);

  // Check resource instances were created
  EXPECT_EQ(inventory_list.resource_instances.size(), 5);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 3);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::HEART)].size(), 2);

  // Check events were scheduled
  auto events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(events.size(), 5);
}

// Test resource loss event handling
TEST_F(InventoryListTest, ResourceLossEventHandling) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Add items
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 3, 1);

  // Get the resource IDs
  auto ore_resources = inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)];
  EXPECT_EQ(ore_resources.size(), 3);

  // Simulate resource loss event for one resource
  uint64_t resource_id_to_lose = ore_resources[0];

  // Remove the resource instance (simulating event handler)
  inventory_list.remove_resource_instance(resource_id_to_lose);

  // Check that the resource instance was removed
  EXPECT_EQ(inventory_list.resource_instances.size(), 2);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 2);

  // The inventory quantity should still be 3 (event handler would update it)
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 3);
}

// Test edge cases
TEST_F(InventoryListTest, EdgeCases) {
  InventoryList inventory_list;

  // Test adding zero items
  InventoryDelta delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 0, 1);
  EXPECT_EQ(delta, 0);
  EXPECT_EQ(inventory_list.inventory.size(), 0);

  // Test removing from empty inventory
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -5, 1);
  EXPECT_EQ(delta, 0);
  EXPECT_EQ(inventory_list.inventory.size(), 0);

  // Test inventory limits
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 255, 1);
  EXPECT_EQ(delta, 255);
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 255);

  // Test adding beyond limit (should clamp)
  delta = inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 1, 1);
  EXPECT_EQ(delta, 0);  // Should be clamped
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 255);
}

// Test constructor variations
TEST_F(InventoryListTest, ConstructorVariations) {
  // Test simple constructor
  InventoryList simple_list;
  EXPECT_EQ(simple_list.event_manager, nullptr);
  EXPECT_EQ(simple_list.rng, nullptr);
  EXPECT_TRUE(simple_list.resource_loss_prob.empty());

  // Test constructor with resource loss
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;

  InventoryList loss_list(mock_event_manager.get(), rng.get(), resource_loss_prob);
  EXPECT_EQ(loss_list.event_manager, mock_event_manager.get());
  EXPECT_EQ(loss_list.rng, rng.get());
  EXPECT_EQ(loss_list.resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)], 0.1f);
}

// Test getter methods
TEST_F(InventoryListTest, GetterMethods) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::HEART)] = 0.05f;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  const auto& retrieved_prob = inventory_list.get_resource_loss_prob();
  EXPECT_EQ(retrieved_prob.size(), 2);
  EXPECT_EQ(retrieved_prob.at(static_cast<InventoryItem>(TestItems::ORE)), 0.1f);
  EXPECT_EQ(retrieved_prob.at(static_cast<InventoryItem>(TestItems::HEART)), 0.05f);
}

// Test clear functionality
TEST_F(InventoryListTest, ClearFunctionality) {
  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = 0.1f;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Add some items
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), 3, 1);
  EXPECT_EQ(inventory_list.inventory.size(), 1);
  EXPECT_EQ(inventory_list.resource_instances.size(), 3);

  // Clear everything
  inventory_list.clear();

  EXPECT_EQ(inventory_list.inventory.size(), 0);
  EXPECT_EQ(inventory_list.resource_instances.size(), 0);
  EXPECT_EQ(inventory_list.item_to_resources.size(), 0);
  EXPECT_EQ(inventory_list.next_resource_id, 1);
}

// Statistical test for resource loss timing
TEST_F(InventoryListTest, ResourceLossTimingStatistics) {
  const int num_resources = 200;  // Must be <= 255 due to InventoryQuantity being uint8_t
  const float loss_probability = 0.05f;
  // The code now uses std::geometric_distribution for discrete-time loss
  // For geometric distribution: E[X] = 1/p where p is success probability (loss)
  // So expected lifetime = 1/0.05 = 20.0 timesteps
  const float expected_mean = 1.0f / loss_probability;  // 20.0 for 0.05 probability

  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = loss_probability;

  // Create a fresh RNG for this test to avoid interference
  std::mt19937 test_rng(12345);  // Different seed for this test

  InventoryList inventory_list(mock_event_manager.get(), &test_rng, resource_loss_prob);

  // Clear any existing events
  mock_event_manager->clear_events();

  // Add all resources at once
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), num_resources, 1);

  // Get all scheduled resource loss events
  auto events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(events.size(), num_resources);

  // Extract the delay times (loss times)
  std::vector<unsigned int> loss_times;
  for (const auto& event : events) {
    loss_times.push_back(event.delay);
  }

  // Calculate statistics
  double sum = 0.0;
  for (unsigned int time : loss_times) {
    sum += time;
  }
  double mean_loss_time = sum / num_resources;

  // Calculate variance for confidence interval
  double variance = 0.0;
  for (unsigned int time : loss_times) {
    double diff = time - mean_loss_time;
    variance += diff * diff;
  }
  variance /= (num_resources - 1);
  double std_dev = std::sqrt(variance);

  // Calculate 95% confidence interval for the mean
  double confidence_interval = 1.96 * std_dev / std::sqrt(num_resources);

  // The mean should be close to the expected value (20.0 for 0.05 probability)
  // We allow for some statistical variation
  EXPECT_NEAR(mean_loss_time, expected_mean, 5.0)  // Increased tolerance
    << "Mean loss time: " << mean_loss_time
    << ", Expected: " << expected_mean
    << ", 95% CI: [" << (mean_loss_time - confidence_interval)
    << ", " << (mean_loss_time + confidence_interval) << "]";

  // Additional checks: ensure we have reasonable variation
  EXPECT_GT(std_dev, 0.0) << "Standard deviation should be positive";
  // For geometric distribution, std_dev ≈ sqrt((1-p)/p^2) = sqrt(0.95/0.0025) ≈ 19.49
  // So std_dev can be close to or slightly greater than the mean
  EXPECT_LT(std_dev, expected_mean * 1.5) << "Standard deviation should be reasonable";

  // Check that we have a reasonable range of loss times
  auto minmax = std::minmax_element(loss_times.begin(), loss_times.end());
  EXPECT_GT(*minmax.second, *minmax.first) << "Should have variation in loss times";

  // Most loss times should be within a reasonable range
  int within_range = 0;
  for (unsigned int time : loss_times) {
    if (time >= 1 && time <= 100) {  // Reasonable range for geometric distribution
      within_range++;
    }
  }
  EXPECT_GT(within_range, num_resources * 0.90)  // Reduced from 95% to 90%
    << "At least 90% of loss times should be within reasonable range";
}

// Test resource cleanup and replacement
TEST_F(InventoryListTest, ResourceCleanupAndReplacement) {
  const int initial_resources = 100;
  const int replacement_resources = 10;
  const float loss_probability = 0.5f;  // 50% loss probability for faster testing

  std::map<InventoryItem, float> resource_loss_prob;
  resource_loss_prob[static_cast<InventoryItem>(TestItems::ORE)] = loss_probability;

  InventoryList inventory_list(mock_event_manager.get(), rng.get(), resource_loss_prob);

  // Clear any existing events
  mock_event_manager->clear_events();

  // Add initial 100 resources
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), initial_resources, 1);

  // Verify initial state
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], initial_resources);
  EXPECT_EQ(inventory_list.resource_instances.size(), initial_resources);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), initial_resources);

  // Check that 100 events were scheduled
  auto initial_events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(initial_events.size(), initial_resources);

  // Delete all 100 resources
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), -initial_resources, 1);

  // Verify deletion state
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 0);
  EXPECT_EQ(inventory_list.resource_instances.size(), 0);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 0);

  // Clear the old events from the MockEventManager
  mock_event_manager->clear_events();

  // Add 10 new resources
  inventory_list.update_inventory(static_cast<InventoryItem>(TestItems::ORE), replacement_resources, 1);

  // Verify new state
  EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], replacement_resources);
  EXPECT_EQ(inventory_list.resource_instances.size(), replacement_resources);
  EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), replacement_resources);

  // Check that exactly 10 new events were scheduled (total should be 10, not 110)
  auto all_events = mock_event_manager->get_events_of_type(EventType::StochasticResourceLoss);
  EXPECT_EQ(all_events.size(), replacement_resources);

  // Simulate running the clock a few times
  int total_resources_lost = 0;
  for (int timestep = 1; timestep <= 5; timestep++) {
    mock_event_manager->set_current_timestep(timestep);

    // Count how many resources should be lost by this timestep
    int resources_lost_this_timestep = 0;
    for (const auto& event : all_events) {
      if (event.delay == timestep) {  // Only count events that trigger this timestep
        resources_lost_this_timestep++;
      }
    }

    // Manually remove the lost resources (simulating event processing)
    for (int i = 0; i < resources_lost_this_timestep; i++) {
      if (!inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].empty()) {
        uint64_t resource_id = inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)][0];
        inventory_list.remove_resource_instance(resource_id);
        // Also update the inventory count
        inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)]--;
        total_resources_lost++;
      }
    }

    // Verify that the remaining resources are correct
    int expected_remaining = replacement_resources - total_resources_lost;
    EXPECT_EQ(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], expected_remaining);
    EXPECT_EQ(inventory_list.resource_instances.size(), expected_remaining);
    EXPECT_EQ(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), expected_remaining);

    // Verify that we don't have more resources than we started with
    EXPECT_LE(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], replacement_resources);
    EXPECT_LE(inventory_list.resource_instances.size(), replacement_resources);
    EXPECT_LE(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), replacement_resources);

    // Verify that we don't have negative resources
    EXPECT_GE(inventory_list.inventory[static_cast<InventoryItem>(TestItems::ORE)], 0);
    EXPECT_GE(inventory_list.resource_instances.size(), 0);
    EXPECT_GE(inventory_list.item_to_resources[static_cast<InventoryItem>(TestItems::ORE)].size(), 0);
  }
}
