
#include "config/observation_features.hpp"

namespace ObservationFeature {
std::shared_ptr<ObservationFeaturesImpl> _instance;

// Define the extern variables
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
ObservationType Vibe;
ObservationType VisitationCounts;
ObservationType Compass;
ObservationType Tag;
ObservationType CooldownRemaining;
ObservationType Clipped;
ObservationType RemainingUses;

void Initialize(const std::unordered_map<std::string, ObservationType>& feature_ids) {
  _instance = std::make_shared<ObservationFeaturesImpl>(feature_ids);

  // Update the global variables with values from the instance
  TypeId = _instance->TypeId;
  Group = _instance->Group;
  Frozen = _instance->Frozen;
  Orientation = _instance->Orientation;
  ReservedForFutureUse = _instance->ReservedForFutureUse;
  ConvertingOrCoolingDown = _instance->ConvertingOrCoolingDown;
  Swappable = _instance->Swappable;
  EpisodeCompletionPct = _instance->EpisodeCompletionPct;
  LastAction = _instance->LastAction;
  LastActionArg = _instance->LastActionArg;
  LastReward = _instance->LastReward;
  Vibe = _instance->Vibe;
  VisitationCounts = _instance->VisitationCounts;
  Compass = _instance->Compass;
  Tag = _instance->Tag;
  CooldownRemaining = _instance->CooldownRemaining;
  Clipped = _instance->Clipped;
  RemainingUses = _instance->RemainingUses;
}
}  // namespace ObservationFeature
