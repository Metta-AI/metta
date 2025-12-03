
#include "config/observation_features.hpp"

namespace ObservationFeature {
std::shared_ptr<ObservationFeaturesImpl> _instance;

// Define the extern variables
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

void Initialize(const std::unordered_map<std::string, ObservationType>& feature_ids) {
  _instance = std::make_shared<ObservationFeaturesImpl>(feature_ids);

  // Update the global variables with values from the instance
  Group = _instance->Group;
  Frozen = _instance->Frozen;
  EpisodeCompletionPct = _instance->EpisodeCompletionPct;
  LastAction = _instance->LastAction;
  LastReward = _instance->LastReward;
  Vibe = _instance->Vibe;
  Compass = _instance->Compass;
  Tag = _instance->Tag;
  CooldownRemaining = _instance->CooldownRemaining;
  Clipped = _instance->Clipped;
  RemainingUses = _instance->RemainingUses;
  Goal = _instance->Goal;
}
}  // namespace ObservationFeature
