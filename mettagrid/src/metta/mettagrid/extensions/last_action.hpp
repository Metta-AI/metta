// extensions/last_action.hpp
#ifndef EXTENSIONS_LAST_ACTION_HPP_
#define EXTENSIONS_LAST_ACTION_HPP_

#include <vector>

#include "extensions/mettagrid_extension.hpp"

class LastAction : public MettaGridExtension {
public:
  void registerObservations(ObservationEncoder* enc) override;
  void onInit(const MettaGrid* env) override;
  void onReset(MettaGrid* env) override;
  void onStep(MettaGrid* env) override;

  std::string getName() const override {
    return "last_action";
  }

  ExtensionStats getStats() const override;

private:
  ObservationType _last_action_feature;
  ObservationType _last_action_arg_feature;
  size_t _num_agents;

  // Store previous actions and action args for each agent
  std::vector<uint8_t> _previous_actions;
  std::vector<uint8_t> _previous_action_args;

  void addLastActionToObservations(MettaGrid* env);
  void updatePreviousActions(const MettaGrid* env);
};

#endif  // EXTENSIONS_LAST_ACTION_HPP_
