#include "rpc/game_registry.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "rpc/proto_converters.hpp"

namespace mettagrid::rpc {

Status GameRegistry::CreateGame(const v1::CreateGameRequest& request) {
  try {
    auto config = ConvertGameConfig(request.config());
    auto map = ConvertMapDefinition(request.map());

    auto engine = std::make_unique<env::MettaGridEngine>(config, map, request.seed());
    const size_t num_agents = engine->num_agents();
    const size_t tokens_per_agent = static_cast<size_t>(config.num_observation_tokens);

    EngineBuffers buffers;
    buffers.num_agents = num_agents;
    buffers.tokens_per_agent = tokens_per_agent;
    buffers.observations.assign(num_agents * tokens_per_agent * 3, 0);
    buffers.terminals = std::make_unique<TerminalType[]>(num_agents);
    buffers.truncations = std::make_unique<TruncationType[]>(num_agents);
    buffers.rewards = std::make_unique<RewardType[]>(num_agents);
    buffers.episode_rewards = std::make_unique<RewardType[]>(num_agents);

    std::fill_n(buffers.terminals.get(), num_agents, false);
    std::fill_n(buffers.truncations.get(), num_agents, false);
    std::fill_n(buffers.rewards.get(), num_agents, 0.0f);
    std::fill_n(buffers.episode_rewards.get(), num_agents, 0.0f);

    engine->set_buffers(buffers.buffer_view());
    engine->reset();

    GameInstance instance;
    instance.engine = std::move(engine);
    instance.buffers = std::move(buffers);
    instance.action_buffer.assign(num_agents * 2, 0);

    std::lock_guard<std::mutex> lock(mutex_);
    if (games_.contains(request.game_id())) {
      return Status::Error("game_id already exists");
    }
    games_.emplace(request.game_id(), std::move(instance));
    return Status::Ok();
  } catch (const std::exception& ex) {
    return Status::Error(ex.what());
  }
}

Status GameRegistry::StepGame(const v1::StepGameRequest& request, v1::StepResult* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = games_.find(request.game_id());
  if (it == games_.end()) {
    return Status::Error("game_id not found");
  }

  GameInstance& game = it->second;
  const size_t num_agents = game.engine->num_agents();

  if (request.flat_actions_size() > 0) {
    if (request.flat_actions_size() != static_cast<int>(num_agents)) {
      return Status::Error("flat_actions length does not match number of agents");
    }
    const auto& flat_map = game.engine->flat_action_map();
    for (size_t i = 0; i < num_agents; ++i) {
      int32_t flat = request.flat_actions(static_cast<int>(i));
      if (flat < 0 || static_cast<size_t>(flat) >= flat_map.size()) {
        game.action_buffer[i * 2] = -1;
        game.action_buffer[i * 2 + 1] = 0;
      } else {
        const auto& mapping = flat_map[static_cast<size_t>(flat)];
        game.action_buffer[i * 2] = mapping.first;
        game.action_buffer[i * 2 + 1] = mapping.second;
      }
    }
  } else if (request.action_types_size() > 0 || request.action_args_size() > 0) {
    if (request.action_types_size() != static_cast<int>(num_agents) ||
        request.action_args_size() != static_cast<int>(num_agents)) {
      return Status::Error("action_types/action_args must match number of agents");
    }
    for (size_t i = 0; i < num_agents; ++i) {
      game.action_buffer[i * 2] = request.action_types(static_cast<int>(i));
      game.action_buffer[i * 2 + 1] = request.action_args(static_cast<int>(i));
    }
  } else {
    return Status::Error("step_game request missing actions");
  }

  env::ActionMatrixView action_view{game.action_buffer.data(), num_agents};
  game.engine->step(action_view);
  PopulateStepResult(game, response);
  return Status::Ok();
}

Status GameRegistry::GetState(const v1::GetStateRequest& request, v1::StateResult* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = games_.find(request.game_id());
  if (it == games_.end()) {
    return Status::Error("game_id not found");
  }
  PopulateStepResult(it->second, response->mutable_snapshot());
  return Status::Ok();
}

Status GameRegistry::DeleteGame(const v1::DeleteGameRequest& request) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t erased = games_.erase(request.game_id());
  if (erased == 0) {
    return Status::Error("game_id not found");
  }
  return Status::Ok();
}

void GameRegistry::PopulateStepResult(const GameInstance& game, v1::StepResult* response) const {
  const size_t num_agents = game.engine->num_agents();
  response->set_current_step(game.engine->current_step);

  response->set_observations(reinterpret_cast<const char*>(game.buffers.observations.data()),
                             game.buffers.observations.size());
  response->set_rewards(reinterpret_cast<const char*>(game.buffers.rewards.get()),
                        sizeof(RewardType) * num_agents);

  std::string terminals_bytes(num_agents, '\0');
  std::string truncations_bytes(num_agents, '\0');
  for (size_t i = 0; i < num_agents; ++i) {
    terminals_bytes[i] = game.buffers.terminals[i] ? '\x01' : '\x00';
    truncations_bytes[i] = game.buffers.truncations[i] ? '\x01' : '\x00';
  }
  response->set_terminals(terminals_bytes);
  response->set_truncations(truncations_bytes);

  const auto& action_success = game.engine->action_success();
  std::string success_bytes(action_success.size(), '\0');
  for (size_t i = 0; i < action_success.size(); ++i) {
    success_bytes[i] = action_success[i] ? '\x01' : '\x00';
  }
  response->set_action_success(success_bytes);
}

}  // namespace mettagrid::rpc
