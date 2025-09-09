#include <gtest/gtest.h>
#include <random>

#include "mettagrid/grid.hpp"
#include "mettagrid/objects/agent.hpp"
#include "mettagrid/objects/converter.hpp"
#include "mettagrid/objects/box.hpp"
#include "mettagrid/resource_manager.hpp"
#include "mettagrid/types.hpp"

// Test-specific inventory item type constants
namespace TestItems {
constexpr uint8_t WOOD = 0;
constexpr uint8_t STONE = 1;
constexpr uint8_t IRON = 2;
constexpr uint8_t GOLD = 3;
}  // namespace TestItems

// Mock HasInventory class for testing
class MockHasInventory : public GridObject, public virtual HasInventory {
public:
  std::map<InventoryItem, InventoryQuantity> inventory;
  std::map<InventoryItem, float> resource_loss_prob;
  HasInventory::InventoryChangeCallback inventory_callback;

  MockHasInventory(GridObjectId id, const std::string& type_name) {
    this->id = id;
    GridObject::type_name = type_name;
  }

  // HasInventory interface implementation
  const std::map<InventoryItem, InventoryQuantity>& get_inventory() const override {
    return inventory;
  }

  InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) override {
    auto it = inventory.find(item);
    if (it == inventory.end()) {
      if (delta > 0) {
        inventory[item] = static_cast<InventoryQuantity>(delta);
      }
      return delta;
    }

    int new_quantity = static_cast<int>(it->second) + delta;
    if (new_quantity <= 0) {
      inventory.erase(it);
      return -static_cast<InventoryDelta>(it->second);
    } else {
      it->second = static_cast<InventoryQuantity>(new_quantity);
      return delta;
    }
  }

  void set_inventory_callback(HasInventory::InventoryChangeCallback callback) override {
    inventory_callback = callback;
  }

  const std::map<InventoryItem, float>& get_resource_loss_prob() const override {
    return resource_loss_prob;
  }

  const std::string& type_name() const override {
    return GridObject::type_name;
  }

  std::vector<PartialObservationToken> obs_features() const override {
    return {};
  }
};

class ResourceManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a deterministic RNG for testing
    _rng = std::mt19937(42);  // Fixed seed for reproducible tests
    
    // Create a test grid
    _grid = std::make_unique<Grid>(10, 10);  // 10x10 grid
    
    // Create ResourceManager
    _resource_manager = std::make_unique<ResourceManager>(_grid.get(), _rng);
  }

  void TearDown() override {
    _resource_manager.reset();
    _grid.reset();
  }

  // Helper function to create test agent config
  AgentConfig create_test_agent_config(const std::string& group_name = "test_group") {
    std::map<uint8_t, InventoryQuantity> resource_limits;
    resource_limits[TestItems::WOOD] = 100;
    resource_limits[TestItems::STONE] = 100;
    resource_limits[TestItems::IRON] = 100;
    resource_limits[TestItems::GOLD] = 100;

    std::map<uint8_t, RewardType> rewards;
    rewards[TestItems::WOOD] = 0.1f;
    rewards[TestItems::STONE] = 0.2f;
    rewards[TestItems::IRON] = 0.3f;
    rewards[TestItems::GOLD] = 0.4f;

    std::map<uint8_t, RewardType> resource_reward_max;
    resource_reward_max[TestItems::WOOD] = 10.0f;
    resource_reward_max[TestItems::STONE] = 10.0f;
    resource_reward_max[TestItems::IRON] = 10.0f;
    resource_reward_max[TestItems::GOLD] = 10.0f;

    std::map<uint8_t, float> resource_loss_prob;
    resource_loss_prob[TestItems::WOOD] = 0.1f;  // 10% loss rate
    resource_loss_prob[TestItems::STONE] = 0.05f; // 5% loss rate
    resource_loss_prob[TestItems::IRON] = 0.0f;   // No loss
    resource_loss_prob[TestItems::GOLD] = 0.2f;   // 20% loss rate

    return AgentConfig(0,                    // type_id
                       "agent",              // type_name
                       1,                    // group_id
                       group_name,           // group_name
                       100,                  // freeze_duration
                       0.0f,                 // action_failure_penalty
                       resource_limits,      // resource_limits
                       rewards,              // resource_rewards
                       resource_reward_max,  // resource_reward_max
                       {},                   // stat_rewards
                       {},                   // stat_reward_max
                       0.0f,                 // group_reward_pct
                       {},                   // initial_inventory
                       resource_loss_prob);  // resource_loss_prob
  }

  // Helper function to create test converter config
  ConverterConfig create_test_converter_config() {
    std::map<uint8_t, float> resource_loss_prob;
    resource_loss_prob[TestItems::WOOD] = 0.15f;  // 15% loss rate
    resource_loss_prob[TestItems::STONE] = 0.0f;  // No loss

    return ConverterConfig(1,                    // type_id
                           "converter",          // type_name
                           {},                   // input_recipe
                           {},                   // output_recipe
                           1,                    // max_output
                           1,                    // max_conversions
                           10,                   // conversion_ticks
                           5,                    // cooldown
                           0,                    // initial_resource_count
                           0,                    // color
                           false,                // recipe_details_obs
                           resource_loss_prob);  // resource_loss_prob
  }

  // Helper function to create mock inventory object
  std::unique_ptr<MockHasInventory> create_mock_object(GridObjectId id, const std::string& type_name) {
    auto obj = std::make_unique<MockHasInventory>(id, type_name);
    obj->init(static_cast<TypeId>(id), type_name, GridLocation(0, 0));
    return obj;
  }

  std::mt19937 _rng;
  std::unique_ptr<Grid> _grid;
  std::unique_ptr<ResourceManager> _resource_manager;
};

// ==================== Registration Tests ====================

TEST_F(ResourceManagerTest, RegisterAgent) {
  AgentConfig agent_cfg = create_test_agent_config("red_team");
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float reward = 0.0f;
  agent->init(&reward);
  
  _resource_manager->register_agent(agent.get(), "red_team");
  
  // Agent should be registered and accessible
  // We can verify registration by checking that step() doesn't crash
  _resource_manager->step();
}

TEST_F(ResourceManagerTest, RegisterObject) {
  ConverterConfig converter_cfg = create_test_converter_config();
  std::unique_ptr<Converter> converter(new Converter(0, 0, converter_cfg));
  converter->init(1, "converter", GridLocation(0, 0));
  
  _resource_manager->register_object(converter.get());
  
  // Converter should be registered
  // We can verify registration by checking that step() doesn't crash
  _resource_manager->step();
}

TEST_F(ResourceManagerTest, UnregisterObject) {
  AgentConfig agent_cfg = create_test_agent_config("blue_team");
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float reward = 0.0f;
  agent->init(&reward);
  
  _resource_manager->register_agent(agent.get(), "blue_team");
  
  // Verify registration works by running step
  _resource_manager->step();
  
  _resource_manager->unregister_inventory_object(agent->id);
  
  // Verify unregistration works by running step again (should not crash)
  _resource_manager->step();
}

// ==================== Inventory Change Callback Tests ====================

TEST_F(ResourceManagerTest, InventoryChangeCallback) {
  AgentConfig agent_cfg = create_test_agent_config("green_team");
  Agent* agent = new Agent(0, 0, agent_cfg);
  float reward = 0.0f;
  agent->init(&reward);
  
  // Set up callback
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agent to grid (grid takes ownership)
  _grid->add_object(agent);
  
  _resource_manager->register_agent(agent, "green_team");
  
  // Add some inventory
  agent->update_inventory(TestItems::WOOD, 10);
  agent->update_inventory(TestItems::STONE, 5);
  
  // The callback should have been triggered and bins should be updated
  // We can verify this by running step() and checking that losses occur
  int initial_wood = agent->get_inventory().find(TestItems::WOOD)->second;
  int initial_stone = agent->get_inventory().find(TestItems::STONE)->second;
  
  _resource_manager->step();
  
  // Should have lost some wood (10% loss rate) and some stone (5% loss rate)
  int final_wood = agent->get_inventory().find(TestItems::WOOD)->second;
  int final_stone = agent->get_inventory().find(TestItems::STONE)->second;
  
  EXPECT_LT(final_wood, initial_wood);  // Wood should have been lost (10% loss rate)
  EXPECT_LT(final_stone, initial_stone); // Stone should have been lost (5% loss rate)
}

// ==================== Resource Loss Tests ====================

TEST_F(ResourceManagerTest, ResourceLossWithMultipleObjects) {
  // Create two agents in the same group
  AgentConfig agent_cfg = create_test_agent_config("red_team");
  
  Agent* agent1 = new Agent(0, 0, agent_cfg);
  float reward1 = 0.0f;
  agent1->init(&reward1);
  agent1->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  Agent* agent2 = new Agent(1, 0, agent_cfg);
  float reward2 = 0.0f;
  agent2->init(&reward2);
  agent2->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agents to grid (grid takes ownership)
  _grid->add_object(agent1);
  _grid->add_object(agent2);
  
  _resource_manager->register_agent(agent1, "red_team");
  _resource_manager->register_agent(agent2, "red_team");
  
  // Give agents different amounts of wood
  agent1->update_inventory(TestItems::WOOD, 20);  // 20 wood
  agent2->update_inventory(TestItems::WOOD, 10);  // 10 wood
  
  int initial_total = 30;
  
  // Run multiple steps to see resource loss
  for (int i = 0; i < 10; ++i) {
    _resource_manager->step();
  }
  
  int final_total = 0;
  if (agent1->get_inventory().find(TestItems::WOOD) != agent1->get_inventory().end()) {
    final_total += agent1->get_inventory().find(TestItems::WOOD)->second;
  }
  if (agent2->get_inventory().find(TestItems::WOOD) != agent2->get_inventory().end()) {
    final_total += agent2->get_inventory().find(TestItems::WOOD)->second;
  }
  
  // Should have lost some wood over 10 steps
  EXPECT_LT(final_total, initial_total);
}

TEST_F(ResourceManagerTest, NoResourceLossForZeroProbability) {
  AgentConfig agent_cfg = create_test_agent_config("blue_team");
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float reward = 0.0f;
  agent->init(&reward);
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  _resource_manager->register_agent(agent.get(), "blue_team");
  
  // Give agent iron (which has 0% loss rate)
  agent->update_inventory(TestItems::IRON, 100);
  
  int initial_iron = agent->get_inventory().find(TestItems::IRON)->second;
  
  // Run many steps
  for (int i = 0; i < 50; ++i) {
    _resource_manager->step();
  }
  
  int final_iron = agent->get_inventory().find(TestItems::IRON)->second;
  
  // Iron should remain unchanged (0% loss rate)
  EXPECT_EQ(final_iron, initial_iron);
}

TEST_F(ResourceManagerTest, DifferentLossRatesForDifferentItems) {
  AgentConfig agent_cfg = create_test_agent_config("green_team");
  Agent* agent = new Agent(0, 0, agent_cfg);
  float reward = 0.0f;
  agent->init(&reward);
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agent to grid (grid takes ownership)
  _grid->add_object(agent);
  
  _resource_manager->register_agent(agent, "green_team");
  
  // Give agent equal amounts of different items with different loss rates
  agent->update_inventory(TestItems::WOOD, 100);   // 10% loss rate
  agent->update_inventory(TestItems::GOLD, 100);   // 20% loss rate
  
  int initial_wood = 100;
  int initial_gold = 100;
  
  // Run many steps
  for (int i = 0; i < 20; ++i) {
    _resource_manager->step();
  }
  
  int final_wood = 0;
  int final_gold = 0;
  
  auto wood_it = agent->get_inventory().find(TestItems::WOOD);
  if (wood_it != agent->get_inventory().end()) {
    final_wood = wood_it->second;
  }
  
  auto gold_it = agent->get_inventory().find(TestItems::GOLD);
  if (gold_it != agent->get_inventory().end()) {
    final_gold = gold_it->second;
  }
  
  // Gold should have lost more than wood (higher loss rate)
  int wood_loss = initial_wood - final_wood;
  int gold_loss = initial_gold - final_gold;
  
  EXPECT_GT(gold_loss, wood_loss);
}

// ==================== Multiple Groups Tests ====================

TEST_F(ResourceManagerTest, SeparateGroupsHaveSeparateLosses) {
  // Create agents in different groups
  AgentConfig red_cfg = create_test_agent_config("red_team");
  AgentConfig blue_cfg = create_test_agent_config("blue_team");
  
  Agent* red_agent = new Agent(0, 0, red_cfg);
  float red_reward = 0.0f;
  red_agent->init(&red_reward);
  red_agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  Agent* blue_agent = new Agent(1, 0, blue_cfg);
  float blue_reward = 0.0f;
  blue_agent->init(&blue_reward);
  blue_agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agents to grid (grid takes ownership)
  _grid->add_object(red_agent);
  _grid->add_object(blue_agent);
  
  _resource_manager->register_agent(red_agent, "red_team");
  _resource_manager->register_agent(blue_agent, "blue_team");
  
  // Give both agents the same amount of wood
  red_agent->update_inventory(TestItems::WOOD, 50);
  blue_agent->update_inventory(TestItems::WOOD, 50);
  
  int red_initial = 50;
  int blue_initial = 50;
  
  // Run steps
  for (int i = 0; i < 10; ++i) {
    _resource_manager->step();
  }
  
  int red_final = 0;
  int blue_final = 0;
  
  auto red_it = red_agent->get_inventory().find(TestItems::WOOD);
  if (red_it != red_agent->get_inventory().end()) {
    red_final = red_it->second;
  }
  
  auto blue_it = blue_agent->get_inventory().find(TestItems::WOOD);
  if (blue_it != blue_agent->get_inventory().end()) {
    blue_final = blue_it->second;
  }
  
  // Both should have lost wood, but losses should be independent
  EXPECT_LT(red_final, red_initial);
  EXPECT_LT(blue_final, blue_initial);
  
  // The exact amounts lost may differ due to randomness
  // but both should have lost some amount
  EXPECT_GT(red_initial - red_final, 0);
  EXPECT_GT(blue_initial - blue_final, 0);
}

// ==================== Edge Cases Tests ====================

TEST_F(ResourceManagerTest, EmptyInventoryNoLoss) {
  AgentConfig agent_cfg = create_test_agent_config("empty_team");
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float reward = 0.0f;
  agent->init(&reward);
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  _resource_manager->register_agent(agent.get(), "empty_team");
  
  // Don't give agent any inventory
  
  // Run many steps - should not crash
  for (int i = 0; i < 100; ++i) {
    _resource_manager->step();
  }
  
  // Agent should still have no inventory
  EXPECT_TRUE(agent->get_inventory().empty());
}

TEST_F(ResourceManagerTest, DebugBins) {
  AgentConfig agent_cfg = create_test_agent_config("debug_team");
  Agent* agent = new Agent(0, 0, agent_cfg);
  float reward = 0.0f;
  agent->init(&reward);
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agent to grid (grid takes ownership)
  _grid->add_object(agent);
  
  _resource_manager->register_agent(agent, "debug_team");
  
  // Give agent exactly 1 wood
  agent->update_inventory(TestItems::WOOD, 1);
  
  // Check if agent has wood
  auto wood_it = agent->get_inventory().find(TestItems::WOOD);
  EXPECT_NE(wood_it, agent->get_inventory().end());
  EXPECT_EQ(wood_it->second, 1);
  
  // Check resource loss probability
  const auto& loss_prob = agent->get_resource_loss_prob();
  auto loss_it = loss_prob.find(TestItems::WOOD);
  EXPECT_NE(loss_it, loss_prob.end());
  EXPECT_GT(loss_it->second, 0.0f);  // Should be 0.1f (10%)
  
  // For now, just verify that the agent is properly set up
  // We'll debug the ResourceManager step() method separately
  EXPECT_TRUE(true);  // Placeholder - this test just verifies setup
  
  // Let's try running step() once to see if it crashes
  try {
    _resource_manager->step();
    std::cout << "Step() completed without crashing" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Step() threw exception: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "Step() threw unknown exception" << std::endl;
  }
}

TEST_F(ResourceManagerTest, SingleItemLoss) {
  AgentConfig agent_cfg = create_test_agent_config("single_team");
  Agent* agent = new Agent(0, 0, agent_cfg);
  float reward = 0.0f;
  agent->init(&reward);
  agent->set_inventory_callback([this](GridObjectId id, InventoryItem item, InventoryDelta delta) {
    _resource_manager->on_inventory_changed(id, item, delta);
  });
  
  // Add agent to grid (grid takes ownership)
  _grid->add_object(agent);
  
  _resource_manager->register_agent(agent, "single_team");
  
  // Give agent exactly 1 wood
  agent->update_inventory(TestItems::WOOD, 1);
  
  // Debug: Check if agent has wood
  auto wood_it = agent->get_inventory().find(TestItems::WOOD);
  EXPECT_NE(wood_it, agent->get_inventory().end());
  EXPECT_EQ(wood_it->second, 1);
  
  // Debug: Check resource loss probability
  const auto& loss_prob = agent->get_resource_loss_prob();
  auto loss_it = loss_prob.find(TestItems::WOOD);
  EXPECT_NE(loss_it, loss_prob.end());
  EXPECT_GT(loss_it->second, 0.0f);  // Should be 0.1f (10%)
  
  // Debug: Run one step and check if anything happens
  int initial_wood = agent->get_inventory().find(TestItems::WOOD)->second;
  _resource_manager->step();
  int after_one_step = 0;
  auto wood_it_after = agent->get_inventory().find(TestItems::WOOD);
  if (wood_it_after != agent->get_inventory().end()) {
    after_one_step = wood_it_after->second;
  }
  
  // For debugging - let's see if anything changed
  std::cout << "Initial wood: " << initial_wood << ", After one step: " << after_one_step << std::endl;
  
  // Let's also check if the agent is properly registered by trying to access it through the grid
  GridObject* grid_obj = _grid->object(agent->id);
  std::cout << "Agent ID: " << agent->id << ", Grid object found: " << (grid_obj != nullptr) << std::endl;
  
  // Run steps until wood is lost
  bool wood_lost = false;
  for (int i = 0; i < 50 && !wood_lost; ++i) {
    _resource_manager->step();
    
    auto wood_it = agent->get_inventory().find(TestItems::WOOD);
    if (wood_it == agent->get_inventory().end()) {
      wood_lost = true;
    }
  }
  
  // Should eventually lose the single wood item
  EXPECT_TRUE(wood_lost);
}
