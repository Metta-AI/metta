#ifndef EXTENSIONS_OBSERVE_LAST_ACTION_HPP_
#define EXTENSIONS_OBSERVE_LAST_ACTION_HPP_

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "extensions/mettagrid_extension.hpp"
#include "mettagrid_c.hpp"
#include "types.hpp"  // For ActionType and ActionArg

class LastAction : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override {
    _last_action_feature = enc->register_feature("last_action");
    _last_action_arg_feature = enc->register_feature("last_action_arg");
  }

  void onInit(const MettaGrid* env, const GameConfig* /*config*/) override {
    _num_agents = env->num_agents();

    // Allocate storage for previous actions
    _previous_actions.resize(_num_agents, 0);
    _previous_action_args.resize(_num_agents, 0);
  }

  void onReset(MettaGrid* env) override {
    // Initialize previous actions to 0 (no action)
    std::fill(_previous_actions.begin(), _previous_actions.end(), 0);
    std::fill(_previous_action_args.begin(), _previous_action_args.end(), 0);

    // Add initial "no action" to observations
    addLastActionToObservations(env);
  }

  void onStep(MettaGrid* env) override {
    // First, write the PREVIOUS actions to observations
    addLastActionToObservations(env);

    // Then update our stored previous actions with the current actions
    updatePreviousActions(env);
  }

  std::string getName() const override {
    return "observe_last_action";
  }

private:
  ObservationType _last_action_feature;
  ObservationType _last_action_arg_feature;
  size_t _num_agents;

  // Store previous actions and action args for each agent
  std::vector<ActionType> _previous_actions;
  std::vector<ActionArg> _previous_action_args;

  void addLastActionToObservations(MettaGrid* env) {
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

  void updatePreviousActions(const MettaGrid* env) {
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
};

REGISTER_EXTENSION("observe_last_action", LastAction)

#endif  // EXTENSIONS_OBSERVE_LAST_ACTION_HPP_
