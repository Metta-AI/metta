#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_GAME_REGISTRY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_GAME_REGISTRY_HPP_

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "env/buffer_views.hpp"
#include "env/mettagrid_engine.hpp"
#include "proto/mettagrid/rpc/v1/mettagrid_service.pb.h"
#include "rpc/status.hpp"

namespace mettagrid::rpc {

struct EngineBuffers {
  size_t num_agents = 0;
  size_t tokens_per_agent = 0;
  std::vector<ObservationType> observations;
  std::unique_ptr<TerminalType[]> terminals;
  std::unique_ptr<TruncationType[]> truncations;
  std::unique_ptr<RewardType[]> rewards;
  std::unique_ptr<RewardType[]> episode_rewards;

  env::BufferSet buffer_view() {
    return env::BufferSet{
        env::ObservationBufferView{observations.data(), num_agents, tokens_per_agent, 3},
        env::ArrayView<TerminalType>{terminals.get(), num_agents},
        env::ArrayView<TruncationType>{truncations.get(), num_agents},
        env::ArrayView<RewardType>{rewards.get(), num_agents},
        env::ArrayView<RewardType>{episode_rewards.get(), num_agents},
    };
  }
};

class GameRegistry {
public:
  Status CreateGame(const v1::CreateGameRequest& request);
  Status StepGame(const v1::StepGameRequest& request, v1::StepResult* response);
  Status GetState(const v1::GetStateRequest& request, v1::StateResult* response);
  Status DeleteGame(const v1::DeleteGameRequest& request);

private:
  struct GameInstance {
    std::unique_ptr<env::MettaGridEngine> engine;
    EngineBuffers buffers;
    std::vector<ActionType> action_buffer;
  };

  void PopulateStepResult(const GameInstance& game, v1::StepResult* response) const;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, GameInstance> games_;
};

}  // namespace mettagrid::rpc

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_GAME_REGISTRY_HPP_
