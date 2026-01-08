#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_OBSERVATION_FEATURES_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_OBSERVATION_FEATURES_HPP_

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "core/types.hpp"

// Runtime lookup class for observation feature IDs
// This replaces the compile-time constants in ObservationFeature namespace
class ObservationFeaturesImpl {
public:
  // Initialize from feature_ids map (name -> id)
  explicit ObservationFeaturesImpl(const std::unordered_map<std::string, ObservationType>& feature_ids)
      : _name_to_id(feature_ids) {
    // Cache commonly used feature IDs (all are always present now)
    _group = get("agent:group");
    _frozen = get("agent:frozen");
    _episode_completion_pct = get("episode_completion_pct");
    _last_action = get("last_action");
    _last_reward = get("last_reward");
    _vibe = get("vibe");
    _compass = get("agent:compass");
    _tag = get("tag");
    _cooldown_remaining = get("cooldown_remaining");
    _clipped = get("clipped");
    _remaining_uses = get("remaining_uses");
    _goal = get("goal");

    // Initialize public members (must be done AFTER private members are set above)
    Group = _group;
    Frozen = _frozen;
    EpisodeCompletionPct = _episode_completion_pct;
    LastAction = _last_action;
    LastReward = _last_reward;
    Vibe = _vibe;
    Compass = _compass;
    Tag = _tag;
    CooldownRemaining = _cooldown_remaining;
    Clipped = _clipped;
    RemainingUses = _remaining_uses;
    Goal = _goal;
  }

  // Get feature ID by name (throws if not found)
  ObservationType get(const std::string& name) const {
    auto it = _name_to_id.find(name);
    if (it == _name_to_id.end()) {
      throw std::runtime_error("Unknown observation feature: " + name);
    }
    return it->second;
  }

  // Check if feature exists
  bool has(const std::string& name) const {
    return _name_to_id.find(name) != _name_to_id.end();
  }

  // Commonly used feature IDs (cached for performance)
  ObservationType Group;
  ObservationType Frozen;
  ObservationType EpisodeCompletionPct;
  ObservationType LastAction;
  ObservationType LastReward;
  ObservationType Vibe;
  ObservationType Compass;
  ObservationType Tag;
  ObservationType CooldownRemaining;
  ObservationType Clipped;
  ObservationType RemainingUses;
  ObservationType Goal;

private:
  std::unordered_map<std::string, ObservationType> _name_to_id;

  // Cached feature IDs
  ObservationType _group;
  ObservationType _frozen;
  ObservationType _episode_completion_pct;
  ObservationType _last_action;
  ObservationType _last_reward;
  ObservationType _vibe;
  ObservationType _compass;
  ObservationType _tag;
  ObservationType _cooldown_remaining;
  ObservationType _clipped;
  ObservationType _remaining_uses;
  ObservationType _goal;
};

// Global singleton instance
// This is initialized by MettaGrid constructor and provides the same syntax as the old namespace
namespace ObservationFeature {
extern std::shared_ptr<ObservationFeaturesImpl> _instance;

// Initialize the singleton (called by MettaGrid)
void Initialize(const std::unordered_map<std::string, ObservationType>& feature_ids);

// Access feature IDs with the same syntax as before: ObservationFeature::X
// These are extern variables defined in observation_features.cpp
extern ObservationType Group;
extern ObservationType Frozen;
extern ObservationType EpisodeCompletionPct;
extern ObservationType LastAction;
extern ObservationType LastReward;
extern ObservationType Vibe;
extern ObservationType Compass;
extern ObservationType Tag;
extern ObservationType CooldownRemaining;
extern ObservationType Clipped;
extern ObservationType RemainingUses;
extern ObservationType Goal;
}  // namespace ObservationFeature

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_OBSERVATION_FEATURES_HPP_
