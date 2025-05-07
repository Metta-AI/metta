#include "action_handler.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

// TestGrid that inherits from Grid
class TestGrid : public Grid {
public:
  // Fix: properly initialize the base class Grid with required parameters
  TestGrid(unsigned int width, unsigned int height, unsigned int num_layers)
      : Grid(width, height, std::vector<Layer>(10, 0)) {
    // The third parameter is typeToLayerMap - we initialize with a default mapping
    // where all type IDs map to layer 0
    this->num_layers = num_layers;
  }

  // Remove override specifiers since these methods might not be virtual in the base class
  GridObject* object_at(const GridLocation& loc) {
    auto it = objects.find(loc.r * 1000 + loc.c * 10 + loc.layer);
    if (it != objects.end()) {
      return it->second;
    }
    return nullptr;
  }

  GridObject* object(GridObjectId id) {
    auto it = object_map.find(id);
    if (it != object_map.end()) {
      return it->second;
    }
    return nullptr;
  }

  void add_object(GridObject* obj) {
    objects[obj->location.r * 1000 + obj->location.c * 10 + obj->location.layer] = obj;
    object_map[obj->id] = obj;
  }

  void move_object(GridObject* obj, unsigned int r, unsigned int c, unsigned int layer = 0) {
    // Remove from old location
    objects.erase(obj->location.r * 1000 + obj->location.c * 10 + obj->location.layer);
    // Update location
    obj->location = GridLocation(r, c, layer);
    // Add to new location
    objects[r * 1000 + c * 10 + layer] = obj;
  }

  std::map<unsigned int, GridObject*> objects;
  std::map<GridObjectId, GridObject*> object_map;
};

// Remove the duplicate standalone functions that were outside any class
// They were erroneously duplicating the TestGrid methods

// Mock GridObject for testing
class TestObject : public GridObject {
public:
  TestObject(unsigned int id, unsigned int type_id, unsigned int r, unsigned int c, unsigned int layer = 0) {
    this->id = id;
    this->_type_id = type_id;
    this->location = GridLocation(r, c, layer);
  }

  void obs(ObsType* observation, const std::vector<unsigned int>& offsets) const {
    observation[offsets[0]] = _type_id;
  }
};

// Mock Agent for testing - use composition instead of inheritance
class TestAgent {
public:
  GridObjectId id;
  GridLocation location;
  float* reward_ptr;
  unsigned int _type_id;

  TestAgent(unsigned int id, unsigned int r, unsigned int c, float* reward_ptr = nullptr) {
    this->id = id;
    this->location = GridLocation(r, c, 0);
    this->reward_ptr = reward_ptr;
  }

  void init(float* reward) {
    reward_ptr = reward;
  }

  // Remove the duplicate declaration of reward_ptr
};

// Concrete implementation of ActionHandler for movement
class MoveActionHandler : public ActionHandler {
public:
  MoveActionHandler(unsigned int priority = 0) {
    this->priority = priority;
    this->_max_arg = 3;  // 0=up, 1=right, 2=down, 3=left
  }

  void init(Grid* grid) {
    _grid = grid;
  }

  unsigned char max_arg() const {
    return _max_arg;
  }

  bool handle_action(unsigned int agent_idx, GridObjectId agent_id, ActionArg arg, unsigned int timestep) {
    Agent* agent = dynamic_cast<Agent*>(_grid->object(agent_id));
    if (!agent) return false;

    unsigned int r = agent->location.r;
    unsigned int c = agent->location.c;
    unsigned int layer = agent->location.layer;

    // Calculate new position based on direction
    switch (arg) {
      case 0:  // Up
        if (r > 0) r--;
        break;
      case 1:  // Right
        if (c < _grid->width - 1) c++;
        break;
      case 2:  // Down
        if (r < _grid->height - 1) r++;
        break;
      case 3:  // Left
        if (c > 0) c--;
        break;
      default:
        return false;
    }

    // Check if the new position is valid (not occupied)
    GridLocation new_loc(r, c, layer);
    if (_grid->object_at(new_loc) != nullptr) {
      return false;
    }

    // Move the agent using our TestGrid
    TestGrid* test_grid = static_cast<TestGrid*>(_grid);
    test_grid->move_object(agent, r, c, layer);
    return true;
  }

  Grid* _grid;
  unsigned char _max_arg;
};

// Concrete implementation of ActionHandler for collecting resources
class CollectActionHandler : public ActionHandler {
public:
  CollectActionHandler(unsigned int priority = 1) {
    this->priority = priority;
    this->_max_arg = 3;  // 0=up, 1=right, 2=down, 3=left
  }

  void init(Grid* grid) {
    _grid = grid;
  }

  unsigned char max_arg() const {
    return _max_arg;
  }

  bool handle_action(unsigned int agent_idx, GridObjectId agent_id, ActionArg arg, unsigned int timestep) {
    Agent* agent = dynamic_cast<Agent*>(_grid->object(agent_id));
    if (!agent) return false;

    unsigned int r = agent->location.r;
    unsigned int c = agent->location.c;

    // Calculate direction based on arg
    unsigned int target_r = r;
    unsigned int target_c = c;
    switch (arg) {
      case 0:  // Up
        if (r > 0) target_r--;
        break;
      case 1:  // Right
        if (c < _grid->width - 1) target_c++;
        break;
      case 2:  // Down
        if (r < _grid->height - 1) target_r++;
        break;
      case 3:  // Left
        if (c > 0) target_c--;
        break;
      default:
        return false;
    }

    // Check if there's a collectible object at the target position
    GridLocation target_loc(target_r, target_c, 0);
    GridObject* target = _grid->object_at(target_loc);
    if (target && target->_type_id == 3) {  // Type 3 = collectible
      // Give reward to agent
      if (agent->reward_ptr) {
        *(agent->reward_ptr) += 1.0f;
      }

      // Remove the collectible object
      TestGrid* test_grid = dynamic_cast<TestGrid*>(_grid);
      test_grid->objects.erase(target_r * 1000 + target_c * 10);
      test_grid->object_map.erase(target->id);

      return true;
    }

    return false;
  }

  Grid* _grid;
  unsigned char _max_arg;
};

// Test fixture for ActionHandler
class ActionHandlerTest : public ::testing::Test {
protected:
  std::unique_ptr<TestGrid> grid;
  std::unique_ptr<MoveActionHandler> move_handler;
  std::unique_ptr<CollectActionHandler> collect_handler;
  TestAgent* agent;
  float agent_reward;

  void SetUp() override {
    // Create a 10x10 grid with 1 layer
    grid = std::make_unique<TestGrid>(10, 10, 1);

    // Create action handlers
    move_handler = std::make_unique<MoveActionHandler>(0);
    collect_handler = std::make_unique<CollectActionHandler>(1);

    // Initialize handlers with the grid
    move_handler->init(grid.get());
    collect_handler->init(grid.get());

    // Create agent and add to grid
    agent_reward = 0.0f;
    agent = new TestAgent(1, 5, 5, &agent_reward);
    grid->add_object(agent);
  }

  void TearDown() override {
    delete agent;
  }
};

// Test move action handler
TEST_F(ActionHandlerTest, MoveAction) {
  // Check initial position
  EXPECT_EQ(5, agent->location.r);
  EXPECT_EQ(5, agent->location.c);

  // Test move up (arg = 0)
  bool success = move_handler->handle_action(0, agent->id, 0, 1);
  EXPECT_TRUE(success);
  EXPECT_EQ(4, agent->location.r);
  EXPECT_EQ(5, agent->location.c);

  // Test move right (arg = 1)
  success = move_handler->handle_action(0, agent->id, 1, 2);
  EXPECT_TRUE(success);
  EXPECT_EQ(4, agent->location.r);
  EXPECT_EQ(6, agent->location.c);

  // Test move down (arg = 2)
  success = move_handler->handle_action(0, agent->id, 2, 3);
  EXPECT_TRUE(success);
  EXPECT_EQ(5, agent->location.r);
  EXPECT_EQ(6, agent->location.c);

  // Test move left (arg = 3)
  success = move_handler->handle_action(0, agent->id, 3, 4);
  EXPECT_TRUE(success);
  EXPECT_EQ(5, agent->location.r);
  EXPECT_EQ(5, agent->location.c);
}

// Test boundary conditions for move action
TEST_F(ActionHandlerTest, MoveBoundary) {
  // Move agent to edge of grid
  grid->move_object(agent, 0, 0, 0);

  // Test move up (should fail due to boundary)
  bool success = move_handler->handle_action(0, agent->id, 0, 1);
  EXPECT_FALSE(success);
  EXPECT_EQ(0, agent->location.r);
  EXPECT_EQ(0, agent->location.c);

  // Test move left (should fail due to boundary)
  success = move_handler->handle_action(0, agent->id, 3, 2);
  EXPECT_FALSE(success);
  EXPECT_EQ(0, agent->location.r);
  EXPECT_EQ(0, agent->location.c);

  // Move agent to opposite edge
  grid->move_object(agent, 9, 9, 0);

  // Test move down (should fail due to boundary)
  success = move_handler->handle_action(0, agent->id, 2, 3);
  EXPECT_FALSE(success);
  EXPECT_EQ(9, agent->location.r);
  EXPECT_EQ(9, agent->location.c);

  // Test move right (should fail due to boundary)
  success = move_handler->handle_action(0, agent->id, 1, 4);
  EXPECT_FALSE(success);
  EXPECT_EQ(9, agent->location.r);
  EXPECT_EQ(9, agent->location.c);
}

// Test collision handling for move action
TEST_F(ActionHandlerTest, MoveCollision) {
  // Create an obstacle
  auto obstacle = new TestObject(2, 2, 5, 6, 0);
  grid->add_object(obstacle);

  // Test move right (should fail due to obstacle)
  bool success = move_handler->handle_action(0, agent->id, 1, 1);
  EXPECT_FALSE(success);
  EXPECT_EQ(5, agent->location.r);
  EXPECT_EQ(5, agent->location.c);

  // Clean up
  delete obstacle;
}

// Test collect action handler
TEST_F(ActionHandlerTest, CollectAction) {
  // Create a collectible object to the right of the agent
  auto collectible = new TestObject(2, 3, 5, 6, 0);
  grid->add_object(collectible);

  // Check initial reward
  EXPECT_FLOAT_EQ(0.0f, agent_reward);

  // Test collect action to the right (arg = 1)
  bool success = collect_handler->handle_action(0, agent->id, 1, 1);
  EXPECT_TRUE(success);

  // Check that reward was given
  EXPECT_FLOAT_EQ(1.0f, agent_reward);

  // Check that collectible was removed
  EXPECT_EQ(nullptr, grid->object_at(GridLocation(5, 6, 0)));
  EXPECT_EQ(nullptr, grid->object(2));
}

// Test collect action with no collectible
TEST_F(ActionHandlerTest, CollectNoTarget) {
  // Create a non-collectible object (type != 3)
  auto non_collectible = new TestObject(2, 2, 5, 6, 0);
  grid->add_object(non_collectible);

  // Test collect action to the right (arg = 1)
  bool success = collect_handler->handle_action(0, agent->id, 1, 1);
  EXPECT_FALSE(success);

  // Check that reward was not given
  EXPECT_FLOAT_EQ(0.0f, agent_reward);

  // Check that object was not removed
  EXPECT_NE(nullptr, grid->object_at(GridLocation(5, 6, 0)));
  EXPECT_NE(nullptr, grid->object(2));

  // Clean up
  delete non_collectible;
}

// Test collect action boundary conditions
TEST_F(ActionHandlerTest, CollectBoundary) {
  // Move agent to edge of grid
  grid->move_object(agent, 0, 0, 0);

  // Test collect up (should fail due to boundary)
  bool success = collect_handler->handle_action(0, agent->id, 0, 1);
  EXPECT_FALSE(success);

  // Test collect left (should fail due to boundary)
  success = collect_handler->handle_action(0, agent->id, 3, 2);
  EXPECT_FALSE(success);

  // Check that reward was not given
  EXPECT_FLOAT_EQ(0.0f, agent_reward);
}

// Test action priority
TEST_F(ActionHandlerTest, ActionPriority) {
  // Create a custom MettaGrid with both action handlers
  std::vector<ActionHandler*> handlers = {move_handler.get(), collect_handler.get()};
  MettaGrid metta_grid(grid.get(), 1, 100, 3, 3);
  metta_grid.init_action_handlers(handlers);

  // Create action array
  int** actions = new int*[1];
  actions[0] = new int[2];
  actions[0][0] = 0;  // Move action
  actions[0][1] = 1;  // Right

  // Set buffers
  ObsType* observations = new ObsType[1 * 3 * 3 * 3];
  char* terminals = new char[1];
  char* truncations = new char[1];
  float* rewards = new float[1];
  float* episode_rewards = new float[1];

  memset(observations, 0, 1 * 3 * 3 * 3 * sizeof(ObsType));
  memset(terminals, 0, 1 * sizeof(char));
  memset(truncations, 0, 1 * sizeof(char));
  memset(rewards, 0, 1 * sizeof(float));
  memset(episode_rewards, 0, 1 * sizeof(float));

  metta_grid.set_buffers(observations, terminals, truncations, rewards, episode_rewards, 1);

  // Add agent to MettaGrid
  metta_grid.add_agent(agent);

  // Add a collectible to the right of the agent
  auto collectible = new TestObject(2, 3, 5, 6, 0);
  grid->add_object(collectible);

  // Step the environment - the higher priority action (move) should execute first
  metta_grid.step(actions);

  // Agent should have moved right
  EXPECT_EQ(5, agent->location.r);
  EXPECT_EQ(6, agent->location.c);

  // Clean up
  delete[] actions[0];
  delete[] actions;
  delete[] observations;
  delete[] terminals;
  delete[] truncations;
  delete[] rewards;
  delete[] episode_rewards;
  delete collectible;
}

// Run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}