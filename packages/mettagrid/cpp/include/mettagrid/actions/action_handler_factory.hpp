#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_FACTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_FACTORY_HPP_

#include <memory>
#include <random>
#include <vector>

#include "actions/action_handler.hpp"

// Forward declarations
class Grid;
struct GameConfig;

struct ActionHandlerResult {
  std::vector<Action> actions;                           // All actions from all handlers
  std::vector<std::unique_ptr<ActionHandler>> handlers;  // Owns the ActionHandler objects
  unsigned char max_priority;
};

// Creates all action handlers based on game config
ActionHandlerResult create_action_handlers(const GameConfig& game_config, Grid* grid, std::mt19937* rng);

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_FACTORY_HPP_
