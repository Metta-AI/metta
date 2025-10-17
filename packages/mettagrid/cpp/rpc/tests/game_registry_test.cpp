#include <gtest/gtest.h>

#include "core/types.hpp"
#include "rpc/game_registry.hpp"
#include "rpc/status.hpp"
#include "mettagrid/rpc/v1/mettagrid_service.pb.h"

namespace {

mettagrid::rpc::v1::CreateGameRequest MakeMinimalRequest() {
  using namespace mettagrid::rpc;
  using mettagrid::rpc::v1::ActionDefinition;

  v1::CreateGameRequest request;
  request.set_game_id("test_env");
  request.set_seed(42);

  auto* cfg = request.mutable_config();
  cfg->set_num_agents(1);
  cfg->set_max_steps(10);
  cfg->set_episode_truncates(false);
  cfg->set_obs_width(5);
  cfg->set_obs_height(5);
  cfg->set_num_observation_tokens(32);

  cfg->mutable_resource_names()->Add("ore");

  auto* global_obs = cfg->mutable_global_obs();
  global_obs->set_episode_completion_pct(true);
  global_obs->set_last_action(true);
  global_obs->set_last_reward(true);
  global_obs->set_visitation_counts(false);

  auto* action = cfg->add_actions();
  action->set_name("noop");
  action->set_type(ActionDefinition::ACTION_NOOP);
  action->mutable_noop();

  auto* agent_object = cfg->add_objects();
  agent_object->set_name("agent.default");
  auto* agent_cfg = agent_object->mutable_agent();
  agent_cfg->set_type_id(1);
  agent_cfg->set_type_name("agent");
  agent_cfg->set_group_id(0);
  agent_cfg->set_group_name("default");
  agent_cfg->set_action_failure_penalty(0.0f);
  agent_cfg->set_freeze_duration(0);
  agent_cfg->set_group_reward_pct(0.0f);
  agent_cfg->mutable_inventory();

  auto* wall_object = cfg->add_objects();
  wall_object->set_name("wall");
  auto* wall_cfg = wall_object->mutable_wall();
  wall_cfg->set_type_id(2);
  wall_cfg->set_type_name("wall");
  wall_cfg->set_swappable(false);

  auto* map = request.mutable_map();
  const uint32_t height = 3;
  const uint32_t width = 3;
  map->set_height(height);
  map->set_width(width);

  for (uint32_t r = 0; r < height; ++r) {
    for (uint32_t c = 0; c < width; ++c) {
      auto* cell = map->add_cells();
      cell->set_row(r);
      cell->set_col(c);
      if (r == 1 && c == 1) {
        cell->set_object_type("agent.default");
      } else {
        cell->set_object_type("wall");
      }
    }
  }

  return request;
}

}  // namespace

TEST(GameRegistryTest, CreateStepDeleteLifecycle) {
  using namespace mettagrid::rpc;

  GameRegistry registry;
  auto create_request = MakeMinimalRequest();
  Status status = registry.CreateGame(create_request);
  ASSERT_TRUE(status.ok) << status.message;

  v1::StepGameRequest step_request;
  step_request.set_game_id("test_env");
  step_request.add_flat_actions(0);

  v1::StepResult step_result;
  status = registry.StepGame(step_request, &step_result);
  ASSERT_TRUE(status.ok) << status.message;
  EXPECT_EQ(step_result.current_step(), 1u);

  const size_t expected_obs_bytes =
      static_cast<size_t>(create_request.config().num_agents()) *
      create_request.config().num_observation_tokens() * 3;
  EXPECT_EQ(step_result.observations().size(), expected_obs_bytes);
  EXPECT_EQ(step_result.rewards().size(), sizeof(RewardType) * create_request.config().num_agents());
  EXPECT_EQ(step_result.terminals().size(), create_request.config().num_agents());
  EXPECT_EQ(step_result.truncations().size(), create_request.config().num_agents());
  EXPECT_EQ(step_result.action_success().size(), create_request.config().num_agents());

  v1::DeleteGameRequest delete_request;
  delete_request.set_game_id("test_env");
  status = registry.DeleteGame(delete_request);
  EXPECT_TRUE(status.ok) << status.message;
}
