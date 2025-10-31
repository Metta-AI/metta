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
    _type_id = get("type_id");
    _group = get("agent:group");
    _frozen = get("agent:frozen");
    _orientation = get("agent:orientation");
    _reserved_for_future_use = get("agent:reserved_for_future_use");
    _converting = get("converting");
    _swappable = get("swappable");
    _episode_completion_pct = get("episode_completion_pct");
    _last_action = get("last_action");
    _last_action_arg = get("last_action_arg");
    _last_reward = get("last_reward");
    _glyph = get("agent:glyph");
    _visitation_counts = get("agent:visitation_counts");
    _tag = get("tag");
    _cooldown_remaining = get("cooldown_remaining");
    _clipped = get("clipped");
    _remaining_uses = get("remaining_uses");

    // Initialize public members (must be done AFTER private members are set above)
    TypeId = _type_id;
    Group = _group;
    Frozen = _frozen;
    Orientation = _orientation;
    ReservedForFutureUse = _reserved_for_future_use;
    ConvertingOrCoolingDown = _converting;
    Swappable = _swappable;
    EpisodeCompletionPct = _episode_completion_pct;
    LastAction = _last_action;
    LastActionArg = _last_action_arg;
    LastReward = _last_reward;
    Glyph = _glyph;
    VisitationCounts = _visitation_counts;
    Tag = _tag;
    CooldownRemaining = _cooldown_remaining;
    Clipped = _clipped;
    RemainingUses = _remaining_uses;
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
  ObservationType TypeId;
  ObservationType Group;
  ObservationType Frozen;
  ObservationType Orientation;
  ObservationType ReservedForFutureUse;
  ObservationType ConvertingOrCoolingDown;
  ObservationType Swappable;
  ObservationType EpisodeCompletionPct;
  ObservationType LastAction;
  ObservationType LastActionArg;
  ObservationType LastReward;
  ObservationType Glyph;
  ObservationType VisitationCounts;
  ObservationType Tag;
  ObservationType CooldownRemaining;
  ObservationType Clipped;
  ObservationType RemainingUses;

private:
  std::unordered_map<std::string, ObservationType> _name_to_id;

  // Cached feature IDs
  ObservationType _type_id;
  ObservationType _group;
  ObservationType _frozen;
  ObservationType _orientation;
  ObservationType _reserved_for_future_use;
  ObservationType _converting;
  ObservationType _swappable;
  ObservationType _episode_completion_pct;
  ObservationType _last_action;
  ObservationType _last_action_arg;
  ObservationType _last_reward;
  ObservationType _glyph;
  ObservationType _visitation_counts;
  ObservationType _tag;
  ObservationType _cooldown_remaining;
  ObservationType _clipped;
  ObservationType _remaining_uses;
};

// Global singleton instance
// This is initialized by MettaGrid constructor and provides the same syntax as the old namespace
namespace ObservationFeature {
extern std::shared_ptr<ObservationFeaturesImpl> _instance;

// Initialize the singleton (called by MettaGrid)
void Initialize(const std::unordered_map<std::string, ObservationType>& feature_ids);

// Access feature IDs with the same syntax as before: ObservationFeature::TypeId
// These are extern variables defined in observation_features.cpp
extern ObservationType TypeId;
extern ObservationType Group;
extern ObservationType Frozen;
extern ObservationType Orientation;
extern ObservationType ReservedForFutureUse;
extern ObservationType ConvertingOrCoolingDown;
extern ObservationType Swappable;
extern ObservationType EpisodeCompletionPct;
extern ObservationType LastAction;
extern ObservationType LastActionArg;
extern ObservationType LastReward;
extern ObservationType Glyph;
extern ObservationType VisitationCounts;
extern ObservationType Tag;
extern ObservationType CooldownRemaining;
extern ObservationType Clipped;
extern ObservationType RemainingUses;
}  // namespace ObservationFeature

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CONFIG_OBSERVATION_FEATURES_HPP_
