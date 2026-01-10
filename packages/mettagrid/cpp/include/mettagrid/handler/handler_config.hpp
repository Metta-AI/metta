#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONFIG_HPP_

#include <string>
#include <variant>
#include <vector>

#include "core/types.hpp"

namespace mettagrid {

// Entity reference for resolving actor/target in filters and mutations
enum class EntityRef {
  actor,             // The object performing the activation
  target,            // The object being activated
  actor_collective,  // The collective of the actor
  target_collective  // The collective of the target
};

// Alignment conditions for AlignmentFilter
enum class AlignmentCondition {
  aligned,              // Actor and target are both aligned (have collectives)
  unaligned,            // Either actor or target has no collective
  same_collective,      // Actor and target belong to same collective
  different_collective  // Actor and target belong to different collectives
};

// Align-to options for AlignmentMutation
enum class AlignTo {
  actor_collective,  // Align target to actor's collective
  none               // Remove target's collective alignment
};

// ============================================================================
// Filter Configs
// ============================================================================

struct VibeFilterConfig {
  EntityRef entity = EntityRef::target;
  ObservationType vibe_id = 0;  // The vibe ID to match (index into vibe_names)
};

struct ResourceFilterConfig {
  EntityRef entity = EntityRef::target;
  InventoryItem resource_id = 0;
  InventoryQuantity min_amount = 1;
};

struct AlignmentFilterConfig {
  AlignmentCondition condition = AlignmentCondition::same_collective;
};

struct TagFilterConfig {
  EntityRef entity = EntityRef::target;
  std::vector<int> required_tag_ids;  // Target must have at least one of these
};

// Variant type for all filter configs
using FilterConfig = std::variant<VibeFilterConfig, ResourceFilterConfig, AlignmentFilterConfig, TagFilterConfig>;

// ============================================================================
// Mutation Configs
// ============================================================================

struct ResourceDeltaMutationConfig {
  EntityRef entity = EntityRef::target;
  InventoryItem resource_id = 0;
  InventoryDelta delta = 0;
};

struct ResourceTransferMutationConfig {
  EntityRef source = EntityRef::actor;
  EntityRef destination = EntityRef::target;
  InventoryItem resource_id = 0;
  InventoryDelta amount = -1;  // -1 means transfer all available
};

struct AlignmentMutationConfig {
  AlignTo align_to = AlignTo::actor_collective;
};

struct FreezeMutationConfig {
  int duration = 1;  // Ticks to freeze
};

struct ClearInventoryMutationConfig {
  EntityRef entity = EntityRef::target;
  // If resource_id is set, only clear that resource; otherwise clear all
  InventoryItem resource_id = 255;  // 255 = clear all
};

struct AttackMutationConfig {
  InventoryItem weapon_resource = 0;
  InventoryItem armor_resource = 0;
  InventoryItem health_resource = 0;
  float damage_multiplier = 1.0f;
};

// Variant type for all mutation configs
using MutationConfig = std::variant<ResourceDeltaMutationConfig,
                                    ResourceTransferMutationConfig,
                                    AlignmentMutationConfig,
                                    FreezeMutationConfig,
                                    ClearInventoryMutationConfig,
                                    AttackMutationConfig>;

// ============================================================================
// Handler Config
// ============================================================================

struct HandlerConfig {
  std::string name;
  std::vector<FilterConfig> filters;      // All must pass for handler to trigger
  std::vector<MutationConfig> mutations;  // Applied sequentially if filters pass

  HandlerConfig() = default;
  explicit HandlerConfig(const std::string& handler_name) : name(handler_name) {}
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_HANDLER_HANDLER_CONFIG_HPP_
