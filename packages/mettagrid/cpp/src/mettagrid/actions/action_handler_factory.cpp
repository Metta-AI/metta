#include "actions/action_handler_factory.hpp"

#include <unordered_map>

#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/move.hpp"
#include "actions/move_config.hpp"
#include "actions/noop.hpp"
#include "actions/transfer.hpp"
#include "config/mettagrid_config.hpp"

ActionHandlerResult create_action_handlers(const GameConfig& game_config, Grid* grid, std::mt19937* rng) {
  ActionHandlerResult result;
  result.max_priority = 0;

  // Noop
  auto noop = std::make_unique<Noop>(*game_config.actions.at("noop"));
  noop->init(grid, rng);
  if (noop->priority > result.max_priority) result.max_priority = noop->priority;
  for (const auto& action : noop->actions()) {
    result.actions.push_back(action);
  }
  result.handlers.push_back(std::move(noop));

  // Move
  auto move_config = std::static_pointer_cast<const MoveActionConfig>(game_config.actions.at("move"));
  auto move = std::make_unique<Move>(*move_config, &game_config);
  move->init(grid, rng);
  if (move->priority > result.max_priority) result.max_priority = move->priority;
  for (const auto& action : move->actions()) {
    result.actions.push_back(action);
  }
  // Capture the raw pointer to pass to other handlers
  Move* move_ptr = move.get();
  result.handlers.push_back(std::move(move));

  // Attack
  auto attack_config = std::static_pointer_cast<const AttackActionConfig>(game_config.actions.at("attack"));
  auto attack = std::make_unique<Attack>(*attack_config, &game_config);
  attack->init(grid, rng);
  if (attack->priority > result.max_priority) result.max_priority = attack->priority;
  for (const auto& action : attack->actions()) {
    result.actions.push_back(action);
  }

  // Transfer
  auto transfer_config = std::static_pointer_cast<const TransferActionConfig>(game_config.actions.at("transfer"));
  auto transfer = std::make_unique<Transfer>(*transfer_config, &game_config);
  transfer->init(grid, rng);
  if (transfer->priority > result.max_priority) result.max_priority = transfer->priority;
  for (const auto& action : transfer->actions()) {
    result.actions.push_back(action);
  }

  // Register vibe-triggered action handlers with Move
  std::unordered_map<std::string, ActionHandler*> handlers;
  handlers["attack"] = attack.get();
  handlers["transfer"] = transfer.get();
  move_ptr->set_action_handlers(handlers);

  result.handlers.push_back(std::move(attack));
  result.handlers.push_back(std::move(transfer));

  // ChangeVibe
  auto change_vibe_config =
      std::static_pointer_cast<const ChangeVibeActionConfig>(game_config.actions.at("change_vibe"));
  auto change_vibe = std::make_unique<ChangeVibe>(*change_vibe_config, &game_config);
  change_vibe->init(grid, rng);
  if (change_vibe->priority > result.max_priority) result.max_priority = change_vibe->priority;
  for (const auto& action : change_vibe->actions()) {
    result.actions.push_back(action);
  }
  result.handlers.push_back(std::move(change_vibe));

  return result;
}
