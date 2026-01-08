#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_ACTIVATION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_ACTIVATION_HANDLER_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

// Filter types for activation handler conditions
enum class FilterType {
  VIBE,       // Check if actor/target has specific vibe
  RESOURCE,   // Check if entity has min resources
  ALIGNMENT,  // Check faction alignment relationship
  TAG         // Check if target has specific tag
};

// Entity selector for filters/mutations
enum class EntitySelector {
  ACTOR,          // The entity performing the action
  TARGET,         // The entity being acted upon
  ACTOR_FACTION,  // The faction of the actor
  TARGET_FACTION  // The faction of the target
};

// Alignment condition types
enum class AlignmentCondition {
  ALIGNED,           // Entity is aligned to any faction
  UNALIGNED,         // Entity is not aligned to any faction
  SAME_FACTION,      // Actor and target are in same faction
  DIFFERENT_FACTION  // Actor and target are in different factions
};

// Filter configuration
struct FilterConfig {
  FilterType type = FilterType::VIBE;
  EntitySelector entity = EntitySelector::ACTOR;

  // VIBE filter: check for specific vibe value
  int vibe_value = 0;

  // RESOURCE filter: check minimum resource amount
  InventoryItem resource_id = 0;
  InventoryDelta min_amount = 0;

  // ALIGNMENT filter: check alignment condition
  AlignmentCondition alignment_condition = AlignmentCondition::ALIGNED;

  // TAG filter: check for specific tag
  int tag_id = 0;

  FilterConfig() = default;

  // Create a vibe filter
  static FilterConfig vibe(EntitySelector entity, int vibe_value) {
    FilterConfig f;
    f.type = FilterType::VIBE;
    f.entity = entity;
    f.vibe_value = vibe_value;
    return f;
  }

  // Create a resource filter
  static FilterConfig resource(EntitySelector entity, InventoryItem resource_id, InventoryDelta min_amount) {
    FilterConfig f;
    f.type = FilterType::RESOURCE;
    f.entity = entity;
    f.resource_id = resource_id;
    f.min_amount = min_amount;
    return f;
  }

  // Create an alignment filter
  static FilterConfig alignment(AlignmentCondition condition) {
    FilterConfig f;
    f.type = FilterType::ALIGNMENT;
    f.alignment_condition = condition;
    return f;
  }

  // Create a tag filter
  static FilterConfig tag(EntitySelector entity, int tag_id) {
    FilterConfig f;
    f.type = FilterType::TAG;
    f.entity = entity;
    f.tag_id = tag_id;
    return f;
  }
};

// Mutation types for activation handler effects
enum class MutationType {
  RESOURCE_DELTA,     // Add/remove resources from entity
  RESOURCE_TRANSFER,  // Transfer resources between entities
  ALIGNMENT,          // Change target's faction alignment
  FREEZE,             // Freeze entity for duration
  ATTACK              // Apply attack/damage
};

// Mutation configuration
struct MutationConfig {
  MutationType type = MutationType::RESOURCE_DELTA;
  EntitySelector entity = EntitySelector::TARGET;

  // RESOURCE_DELTA: resource changes to apply
  std::unordered_map<InventoryItem, InventoryDelta> resource_deltas;

  // RESOURCE_TRANSFER: source, target, and amounts
  EntitySelector transfer_source = EntitySelector::ACTOR;
  EntitySelector transfer_target = EntitySelector::TARGET;

  // ALIGNMENT: align target to actor's faction (true) or unalign (false)
  bool align_to_actor = true;

  // FREEZE: duration in ticks
  int freeze_duration = 0;

  // ATTACK: damage amount (uses existing attack system if present)
  int attack_damage = 0;

  MutationConfig() = default;

  // Create a resource delta mutation
  static MutationConfig resource_delta(EntitySelector entity,
                                       const std::unordered_map<InventoryItem, InventoryDelta>& deltas) {
    MutationConfig m;
    m.type = MutationType::RESOURCE_DELTA;
    m.entity = entity;
    m.resource_deltas = deltas;
    return m;
  }

  // Create a resource transfer mutation
  static MutationConfig resource_transfer(EntitySelector source,
                                          EntitySelector target,
                                          const std::unordered_map<InventoryItem, InventoryDelta>& deltas) {
    MutationConfig m;
    m.type = MutationType::RESOURCE_TRANSFER;
    m.transfer_source = source;
    m.transfer_target = target;
    m.resource_deltas = deltas;
    return m;
  }

  // Create an alignment mutation
  static MutationConfig alignment(bool align_to_actor) {
    MutationConfig m;
    m.type = MutationType::ALIGNMENT;
    m.align_to_actor = align_to_actor;
    return m;
  }

  // Create a freeze mutation
  static MutationConfig freeze(int duration) {
    MutationConfig m;
    m.type = MutationType::FREEZE;
    m.freeze_duration = duration;
    return m;
  }

  // Create an attack mutation
  static MutationConfig attack(int damage) {
    MutationConfig m;
    m.type = MutationType::ATTACK;
    m.attack_damage = damage;
    return m;
  }
};

// Activation handler: a list of filters (all must pass) and mutations (all applied if filters pass)
struct ActivationHandlerConfig {
  std::string name;                       // Handler name for stats tracking
  std::vector<FilterConfig> filters;      // Conditions that must all be true
  std::vector<MutationConfig> mutations;  // Effects to apply when all filters pass

  ActivationHandlerConfig() = default;
  ActivationHandlerConfig(const std::string& name,
                          const std::vector<FilterConfig>& filters,
                          const std::vector<MutationConfig>& mutations)
      : name(name), filters(filters), mutations(mutations) {}
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_ACTIVATION_HANDLER_HPP_
