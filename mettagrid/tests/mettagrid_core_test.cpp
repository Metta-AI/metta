#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "action_handler.hpp"
#include "core.hpp"
#include "event.hpp"
#include "grid.hpp"
#include "objects/agent.hpp"
#include "observation_encoder.hpp"

// Mock classes for testing
class MockGrid : public Grid {
public:
  MockGrid(unsigned int width, unsigned int height, unsigned int num_layers) {
    this->width = width;
    this->height = height;
    this->num_layers = num_layers;
  }

  GridObject* object_at(const GridLocation& loc) override {
    auto it = objects.find(loc.r * 1000 + loc.c * 10 + loc.layer);
    if (it != objects.end()) {
      return it->second;
    }
    return nullptr;
  }

  GridObject* object(GridObjectId id) override {
    auto it = object_map.find(id);
    if (it != object_map.end()) {
      return it->second;
    }
    return nullptr;
  }

  void add_object(GridObject* obj, unsigned int r, unsigned int c, unsigned int layer = 0) {
    obj->location = GridLocation(r, c, layer);
    objects[r * 1000 + c * 10 + layer] = obj;
    object_map[obj->id] = obj;
  }

  std::map<unsigned int, GridObject*> objects;
  std::map<GridObjectId, GridObject*> object_map;
};

class MockAgent : public Agent {
public:
  MockAgent(unsigned int id, unsigned int r, unsigned int c) {
    this->id = id;
    this->location = GridLocation(r, c, 0);
    reward_ptr = nullptr;
  }

  void init(float* reward) override {
    reward_ptr = reward;
  }

  float* reward_ptr;
};

class MockActionHandler : public ActionHandler {
public:
  MockActionHandler(unsigned int priority, unsigned char max_arg_val) : called(false) {
    this->priority = priority;
    this->_max_arg = max_arg_val;
  }

  void init(Grid* grid) override {
    _grid = grid;
  }

  unsigned char max_arg() const override {
    return _max_arg;
  }

  bool handle_action(unsigned int agent_idx, GridObjectId agent_id, ActionArg arg, unsigned int timestep) override {
    called = true;
    last_agent_idx = agent_idx;
    last_agent_id = agent_id;
    last_arg = arg;
    last_timestep = timestep;
    return action_success;
  }

  unsigned char _max_arg;
  Grid* _grid;
  bool called;
  unsigned int last_agent_idx;
  GridObjectId last_agent_id;
  ActionArg last_arg;
  unsigned int last_timestep;
  bool action_success = true;
};

class MockObservationEncoder : public ObservationEncoder {
public:
  std::vector<std::string> feature_names() const override {
    return {"feature1", "feature2", "feature3"};
  }

  void encode(const GridObject* obj, ObsType* observation) override {
    observation[0] = obj->_type_id;
    observation[1] = 42;
    observation[2] = 99;
  }
};

class MockGridObject : public GridObject {
public:
  MockGridObject(unsigned int id, unsigned int type_id, unsigned int r, unsigned int c, unsigned int layer = 0) {
    this->id = id;
    this->_type_id = type_id;
    this->location = GridLocation(r, c, layer);
  }

  void obs(ObsType* observation, const std::vector<unsigned int>& offsets) const override {
    observation[offsets[0]] = _type_id;
  }
};

class MettaGridTest : public ::testing::Test {
protected:
  std::unique_ptr<MockGrid> grid;
  std::unique_ptr<MettaGrid> metta_grid;
  std::vector<MockActionHandler*> action_handlers;
  std::vector<MockAgent*> agents;
  ObsType* observations;
  char* terminals;
  char* truncations;
  float* rewards;
  float* episode_rewards;
  double* group_rewards;
  int** actions;

  void SetUp() override {
    // Create a 10x10 grid with 3 layers
    grid = std::make_unique<MockGrid>(10, 10, 3);

    // Create MettaGrid with 2 agents, 100 max timesteps, and 5x5 observation window
    metta_grid = std::make_unique<MettaGrid>(grid.get(), 2, 100, 5, 5);

    // Replace the default ObservationEncoder with our mock
    metta_grid->init_event_manager(new EventManager());

    // Create action handlers
    action_handlers.push_back(new MockActionHandler(0, 4));  // Priority 0, max arg 4
    action_handlers.push_back(new MockActionHandler(1, 2));  // Priority 1, max arg 2
    action_handlers.push_back(new MockActionHandler(2, 0));  // Priority 2, max arg 0
    metta_grid->init_action_handlers(std::vector<ActionHandler*>(action_handlers.begin(), action_handlers.end()));

    // Allocate memory for buffers
    observations = new ObsType[2 * 5 * 5 * 3];  // 2 agents, 5x5 obs, 3 features
    terminals = new char[2];
    truncations = new char[2];
    rewards = new float[2];
    episode_rewards = new float[2];
    group_rewards = new double[2];

    // Set the buffers
    metta_grid->set_buffers(observations, terminals, truncations, rewards, episode_rewards, 2);
    metta_grid->init_group_rewards(group_rewards, 2);

    // Initialize all buffers to zero
    memset(observations, 0, 2 * 5 * 5 * 3 * sizeof(ObsType));
    memset(terminals, 0, 2 * sizeof(char));
    memset(truncations, 0, 2 * sizeof(char));
    memset(rewards, 0, 2 * sizeof(float));
    memset(episode_rewards, 0, 2 * sizeof(float));
    memset(group_rewards, 0, 2 * sizeof(double));

    // Create agents and add them to the grid
    agents.push_back(new MockAgent(0, 5, 5));  // Agent 0 at (5,5)
    agents.push_back(new MockAgent(1, 7, 7));  // Agent 1 at (7,7)

    // Add agents to MettaGrid
    metta_grid->add_agent(agents[0]);
    metta_grid->add_agent(agents[1]);

    // Create action array
    actions = new int*[2];
    for (int i = 0; i < 2; i++) {
      actions[i] = new int[2];
      actions[i][0] = 0;  // Action type 0
      actions[i][1] = 0;  // Arg 0
    }
  }

  void TearDown() override {
    // Clean up allocated memory
    for (int i = 0; i < 2; i++) {
      delete[] actions[i];
    }
    delete[] actions;
    delete[] observations;
    delete[] terminals;
    delete[] truncations;
    delete[] rewards;
    delete[] episode_rewards;
    delete[] group_rewards;

    for (auto handler : action_handlers) {
      delete handler;
    }

    for (auto agent : agents) {
      delete agent;
    }
  }
};

// Test constructor and initialization
TEST_F(MettaGridTest, Initialization) {
  EXPECT_EQ(10, metta_grid->map_width());
  EXPECT_EQ(10, metta_grid->map_height());
  EXPECT_EQ(0, metta_grid->current_timestep());
  EXPECT_EQ(2, metta_grid->num_agents());

  auto max_args = metta_grid->max_action_args();
  EXPECT_EQ(3, max_args.size());
  EXPECT_EQ(4, max_args[0]);
  EXPECT_EQ(2, max_args[1]);
  EXPECT_EQ(0, max_args[2]);
}

// Test step function with valid actions
TEST_F(MettaGridTest, Step) {
  // Set up actions
  actions[0][0] = 0;  // First agent: action 0
  actions[0][1] = 1;  // Arg 1
  actions[1][0] = 1;  // Second agent: action 1
  actions[1][1] = 2;  // Arg 2

  // Execute step
  metta_grid->step(actions);

  // Check timestep incremented
  EXPECT_EQ(1, metta_grid->current_timestep());

  // Check action handlers were called
  EXPECT_TRUE(action_handlers[0]->called);
  EXPECT_TRUE(action_handlers[1]->called);

  // Check action handler parameters
  EXPECT_EQ(0, action_handlers[0]->last_agent_idx);
  EXPECT_EQ(0, action_handlers[0]->last_agent_id);
  EXPECT_EQ(1, action_handlers[0]->last_arg);
  EXPECT_EQ(1, action_handlers[0]->last_timestep);

  EXPECT_EQ(1, action_handlers[1]->last_agent_idx);
  EXPECT_EQ(1, action_handlers[1]->last_agent_id);
  EXPECT_EQ(2, action_handlers[1]->last_arg);
  EXPECT_EQ(1, action_handlers[1]->last_timestep);
}

// Test step function with invalid actions
TEST_F(MettaGridTest, InvalidActions) {
  // Set up invalid actions
  actions[0][0] = 10;  // Invalid action type
  actions[1][0] = 1;   // Valid action type
  actions[1][1] = 10;  // Invalid arg (too large)

  // Execute step
  metta_grid->step(actions);

  // Check action handlers were not called
  EXPECT_FALSE(action_handlers[0]->called);

  // The second action handler should not be called due to invalid arg
  EXPECT_FALSE(action_handlers[1]->called);
}

// Test observation computation
TEST_F(MettaGridTest, ComputeObservation) {
  // Add some objects to the grid
  auto obj1 = new MockGridObject(1, 1, 5, 5);
  auto obj2 = new MockGridObject(2, 2, 6, 6);
  grid->add_object(obj1, 5, 5, 0);
  grid->add_object(obj2, 6, 6, 1);

  // Set up the observation encoder
  auto obs_encoder = new MockObservationEncoder();
  delete metta_grid->_obs_encoder;  // Delete the default one
  metta_grid->_obs_encoder = obs_encoder;

  // Compute observation from agent 0's position
  ObsType* agent_obs = new ObsType[5 * 5 * 3];  // 5x5 obs, 3 features
  memset(agent_obs, 0, 5 * 5 * 3 * sizeof(ObsType));
  metta_grid->compute_observation(5, 5, 5, 5, agent_obs);

  // Check that objects were observed correctly
  // Center of the observation (2,2) should have obj1
  EXPECT_EQ(1, agent_obs[(2 * 5 + 2) * 3 + 0]);   // Type ID
  EXPECT_EQ(42, agent_obs[(2 * 5 + 2) * 3 + 1]);  // Feature 2
  EXPECT_EQ(99, agent_obs[(2 * 5 + 2) * 3 + 2]);  // Feature 3

  // obj2 should be at (3,3) in the observation
  EXPECT_EQ(2, agent_obs[(3 * 5 + 3) * 3 + 0]);   // Type ID
  EXPECT_EQ(42, agent_obs[(3 * 5 + 3) * 3 + 1]);  // Feature 2
  EXPECT_EQ(99, agent_obs[(3 * 5 + 3) * 3 + 2]);  // Feature 3

  delete[] agent_obs;
  delete obj1;
  delete obj2;
}

// Test reward decay
TEST_F(MettaGridTest, RewardDecay) {
  // Enable reward decay with custom time steps
  metta_grid->enable_reward_decay(50);

  // Set rewards for agents
  rewards[0] = 10.0f;
  rewards[1] = 5.0f;

  // Step the environment
  metta_grid->step(actions);

  // Check that rewards were decayed
  // Decay factor should be 3.0/50 = 0.06
  // Reward multiplier after one step should be 1.0 * (1.0 - 0.06) = 0.94
  EXPECT_NEAR(10.0f * 0.94f, rewards[0], 0.01f);
  EXPECT_NEAR(5.0f * 0.94f, rewards[1], 0.01f);

  // Check that episode rewards were updated
  EXPECT_NEAR(10.0f * 0.94f, episode_rewards[0], 0.01f);
  EXPECT_NEAR(5.0f * 0.94f, episode_rewards[1], 0.01f);

  // Disable reward decay
  metta_grid->disable_reward_decay();

  // Reset rewards
  rewards[0] = 10.0f;
  rewards[1] = 5.0f;

  // Step again
  metta_grid->step(actions);

  // Check that rewards were not decayed
  EXPECT_EQ(10.0f, rewards[0]);
  EXPECT_EQ(5.0f, rewards[1]);
}

// Test group rewards
TEST_F(MettaGridTest, GroupRewards) {
  // Set group IDs for agents
  agents[0]->group = 0;
  agents[1]->group = 0;

  // Set group sizes and percentages
  metta_grid->set_group_size(0, 2);
  metta_grid->set_group_reward_pct(0, 0.5f);  // 50% of rewards go to group pool

  // Set individual rewards
  rewards[0] = 10.0f;
  rewards[1] = 0.0f;

  // Compute group rewards
  metta_grid->compute_group_rewards(rewards);

  // Check results
  // Agent 0: 10.0 * 0.5 = 5.0 to individual, 5.0 to group pool
  // Group pool: 5.0 / 2 = 2.5 per agent
  // Final rewards: Agent 0: 5.0 + 2.5 = 7.5, Agent 1: 0.0 + 2.5 = 2.5
  EXPECT_NEAR(7.5f, rewards[0], 0.01f);
  EXPECT_NEAR(2.5f, rewards[1], 0.01f);
}

// Test truncation at max timestep
TEST_F(MettaGridTest, MaxTimestep) {
  // Step 99 times (to reach timestep 99)
  for (int i = 0; i < 99; i++) {
    metta_grid->step(actions);
  }

  // Check that truncations are still 0
  EXPECT_EQ(0, truncations[0]);
  EXPECT_EQ(0, truncations[1]);

  // Step one more time (to reach timestep 100)
  metta_grid->step(actions);

  // Check that truncations are now 1
  EXPECT_EQ(1, truncations[0]);
  EXPECT_EQ(1, truncations[1]);
}

// Test observation from object perspective
TEST_F(MettaGridTest, ObserveFromObject) {
  // Add an object to the grid
  auto obj = new MockGridObject(1, 1, 3, 3);
  grid->add_object(obj, 3, 3, 0);

  // Set up the observation encoder
  auto obs_encoder = new MockObservationEncoder();
  delete metta_grid->_obs_encoder;  // Delete the default one
  metta_grid->_obs_encoder = obs_encoder;

  // Observe from the object's perspective
  ObsType* obj_obs = new ObsType[5 * 5 * 3];  // 5x5 obs, 3 features
  memset(obj_obs, 0, 5 * 5 * 3 * sizeof(ObsType));
  metta_grid->observe(1, 5, 5, obj_obs);

  // Check that the object was observed correctly at the center
  EXPECT_EQ(1, obj_obs[(2 * 5 + 2) * 3 + 0]);   // Type ID
  EXPECT_EQ(42, obj_obs[(2 * 5 + 2) * 3 + 1]);  // Feature 2
  EXPECT_EQ(99, obj_obs[(2 * 5 + 2) * 3 + 2]);  // Feature 3

  delete[] obj_obs;
  delete obj;
}

// Test observation at specific location
TEST_F(MettaGridTest, ObserveAtLocation) {
  // Add objects to the grid
  auto obj1 = new MockGridObject(1, 1, 2, 2);
  auto obj2 = new MockGridObject(2, 2, 4, 4);
  grid->add_object(obj1, 2, 2, 0);
  grid->add_object(obj2, 4, 4, 1);

  // Set up the observation encoder
  auto obs_encoder = new MockObservationEncoder();
  delete metta_grid->_obs_encoder;  // Delete the default one
  metta_grid->_obs_encoder = obs_encoder;

  // Observe from a specific location (3,3)
  ObsType* loc_obs = new ObsType[5 * 5 * 3];  // 5x5 obs, 3 features
  memset(loc_obs, 0, 5 * 5 * 3 * sizeof(ObsType));
  metta_grid->observe_at(3, 3, 5, 5, loc_obs);

  // obj1 should be at (1,1) in the observation
  EXPECT_EQ(1, loc_obs[(1 * 5 + 1) * 3 + 0]);   // Type ID
  EXPECT_EQ(42, loc_obs[(1 * 5 + 1) * 3 + 1]);  // Feature 2
  EXPECT_EQ(99, loc_obs[(1 * 5 + 1) * 3 + 2]);  // Feature 3

  // obj2 should be at (3,3) in the observation
  EXPECT_EQ(2, loc_obs[(3 * 5 + 3) * 3 + 0]);   // Type ID
  EXPECT_EQ(42, loc_obs[(3 * 5 + 3) * 3 + 1]);  // Feature 2
  EXPECT_EQ(99, loc_obs[(3 * 5 + 3) * 3 + 2]);  // Feature 3

  delete[] loc_obs;
  delete obj1;
  delete obj2;
}

// Test action success tracking
TEST_F(MettaGridTest, ActionSuccess) {
  // Set action success flags
  action_handlers[0]->action_success = true;
  action_handlers[1]->action_success = false;

  // Set up actions
  actions[0][0] = 0;  // First agent: action 0 (will succeed)
  actions[1][0] = 1;  // Second agent: action 1 (will fail)

  // Execute step
  metta_grid->step(actions);

  // Check action success flags
  auto success = metta_grid->action_success();
  EXPECT_TRUE(success[0]);
  EXPECT_FALSE(success[1]);
}

// Run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TEST();
}