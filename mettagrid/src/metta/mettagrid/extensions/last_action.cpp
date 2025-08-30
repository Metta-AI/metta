// extensions/last_action.cpp
#include "extensions/last_action.hpp"

#include <algorithm>
#include <unordered_map>

#include "mettagrid_c.hpp"

void LastAction::registerObservations(ObservationEncoder* enc) {
  _last_action_feature = enc->register_feature("last_action");
  _last_action_arg_feature = enc->register_feature("last_action_arg");
}

void LastAction::onInit(const MettaGrid* env) {
  _num_agents = env->num_agents();

  // Allocate storage for previous actions
  _previous_actions.resize(_num_agents, 0);
  _previous_action_args.resize(_num_agents, 0);
}

void LastAction::onReset(MettaGrid* env) {
  // Initialize previous actions to 0 (no action)
  std::fill(_previous_actions.begin(), _previous_actions.end(), 0);
  std::fill(_previous_action_args.begin(), _previous_action_args.end(), 0);

  // Add initial "no action" to observations
  addLastActionToObservations(env);
}

void LastAction::onStep(MettaGrid* env) {
  // First, write the PREVIOUS actions to observations
  addLastActionToObservations(env);

  // Then update our stored previous actions with the current actions
  updatePreviousActions(env);
}

void LastAction::addLastActionToObservations(MettaGrid* env) {
  // Write previous action and action_arg as global observations for each agent
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    // Create feature and value vectors for global observation
    std::vector<ObservationType> features = {_last_action_feature, _last_action_arg_feature};

    // Cast to ObservationType (uint8_t) for observations
    // Note: This will truncate values > 255, using bitwise AND to make truncation explicit
    std::vector<ObservationType> values = {static_cast<ObservationType>(_previous_actions[agent_idx] & 0xFF),
                                           static_cast<ObservationType>(_previous_action_args[agent_idx] & 0xFF)};

    // Write both observations at once
    writeGlobalObservations(env, agent_idx, features, values);
  }
}

void LastAction::updatePreviousActions(const MettaGrid* env) {
  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    // Get current action and action_arg for this agent
    auto actions = getAgentActions(env, agent_idx);
    if (actions.size() >= 2) {
      // Direct assignment - no conversion needed since types match
      _previous_actions[agent_idx] = actions[0];      // action
      _previous_action_args[agent_idx] = actions[1];  // action_arg
    }
  }
}

ExtensionStats LastAction::getStats() const {
  ExtensionStats stats;

  // Use unordered_map for counting since action space might be larger than 256
  std::unordered_map<ActionType, int> action_counts;
  std::unordered_map<ActionArg, int> arg_counts;

  for (size_t i = 0; i < _num_agents; i++) {
    action_counts[_previous_actions[i]]++;
    arg_counts[_previous_action_args[i]]++;
  }

  // Find most common action
  int max_count = 0;
  ActionType most_common_action = 0;
  for (const auto& [action, count] : action_counts) {
    if (count > max_count) {
      max_count = count;
      most_common_action = action;
    }
  }

  stats["most_common_action"] = static_cast<float>(most_common_action);
  stats["most_common_action_count"] = static_cast<float>(max_count);
  stats["num_unique_actions"] = static_cast<float>(action_counts.size());

  return stats;
}

REGISTER_EXTENSION("last_action", LastAction)
