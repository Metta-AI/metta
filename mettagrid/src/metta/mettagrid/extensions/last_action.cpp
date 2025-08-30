// extensions/last_action.cpp
#include "extensions/last_action.hpp"

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
    std::vector<ObservationType> values = {static_cast<ObservationType>(_previous_actions[agent_idx]),
                                           static_cast<ObservationType>(_previous_action_args[agent_idx])};

    // Write both observations at once
    writeGlobalObservations(env, agent_idx, features, values);
  }
}

void LastAction::updatePreviousActions(const MettaGrid* env) {
  // NOTE: This assumes we have an accessor method to get the current actions
  // You'll need to add something like getAgentActions() to the base class
  // For now, I'll show the pattern assuming such a method exists:

  for (size_t agent_idx = 0; agent_idx < _num_agents; agent_idx++) {
    // Get current action and action_arg for this agent
    // This is pseudo-code - you'll need to implement the actual accessor
    auto actions = getAgentActions(env, agent_idx);
    if (actions.size() >= 2) {
      _previous_actions[agent_idx] = actions[0];      // action
      _previous_action_args[agent_idx] = actions[1];  // action_arg
    }
  }
}

ExtensionStats LastAction::getStats() const {
  ExtensionStats stats;

  // Calculate some basic statistics about action distribution
  std::vector<int> action_counts(256, 0);
  std::vector<int> arg_counts(256, 0);

  for (size_t i = 0; i < _num_agents; i++) {
    action_counts[_previous_actions[i]]++;
    arg_counts[_previous_action_args[i]]++;
  }

  // Find most common action
  int max_count = 0;
  uint8_t most_common_action = 0;
  for (size_t i = 0; i < 256; i++) {
    if (action_counts[i] > max_count) {
      max_count = action_counts[i];
      most_common_action = static_cast<uint8_t>(i);
    }
  }

  stats["most_common_action"] = static_cast<float>(most_common_action);
  stats["most_common_action_count"] = static_cast<float>(max_count);
  stats["num_unique_actions"] =
      static_cast<float>(std::count_if(action_counts.begin(), action_counts.end(), [](int c) { return c > 0; }));

  return stats;
}

REGISTER_EXTENSION("last_action", LastAction)
