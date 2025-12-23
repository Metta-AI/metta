#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "actions/build_config.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"

struct GameConfig;
class Agent;
class ObservationEncoder;
class StatsTracker;
struct GridLocation;

class Build : public ActionHandler {
public:
  explicit Build(const BuildActionConfig& cfg,
                 const GameConfig* game_config,
                 StatsTracker* stats_tracker = nullptr,
                 const std::string& action_name = "build");

  // Set runtime context needed for assembler initialization
  void set_runtime_context(unsigned int* current_timestep_ptr,
                           ObservationEncoder* obs_encoder,
                           unsigned int num_agents);

  std::vector<Action> create_actions() override;

  // Get vibes that trigger this action on move
  const std::vector<ObservationType>& get_vibes() const;

  // Check if the actor's vibe has a build configured
  bool has_build_for_vibe(ObservationType vibe) const;

  // Try to build after a successful move. Returns true if build succeeded.
  // previous_location: where the agent was before moving (where we'll place the object)
  bool try_build(Agent& actor, const GridLocation& previous_location);

protected:
  std::unordered_map<ObservationType, VibeBuildEffect> _vibe_builds;
  bool _enabled;
  std::vector<ObservationType> _vibes;
  const GameConfig* _game_config;
  StatsTracker* _stats_tracker;
  // Runtime context for assembler initialization
  unsigned int* _current_timestep_ptr;
  ObservationEncoder* _obs_encoder;
  unsigned int _num_agents;

  bool _handle_action(Agent& actor, ActionArg arg) override;

private:
  std::string _action_prefix(const std::string& group) const;
  void _log_build_cost(Agent& actor, InventoryItem item, InventoryDelta amount) const;
  GridObject* _create_object(const std::string& object_key, const GridLocation& location);
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_
