#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "actions/activation_handler_config.hpp"
#include "core/types.hpp"

// Forward declarations - these avoid circular dependencies
class Grid;
class GridObject;
class Agent;
class Commons;
class Alignable;
class HasInventory;
class Inventory;
struct GameConfig;

// ===== Activation Context =====
// Provides access to actor, target, and their commons during activation

struct ActivationContext {
  Agent& actor;
  GridObject& target;
  Grid* grid;
  const GameConfig* game_config;

  ActivationContext(Agent& actor, GridObject& target, Grid* grid, const GameConfig* game_config)
      : actor(actor), target(target), grid(grid), game_config(game_config) {}

  // Resolve inventory target
  Inventory* resolve_inventory(const std::string& target_str) const;

  // Resolve alignable target (for alignment mutations)
  Alignable* resolve_alignable(const std::string& target_str) const;

  // Get commons for a target
  Commons* resolve_commons(const std::string& target_str) const;

  // Get agent for a target (returns nullptr if target is not an agent)
  Agent* resolve_agent(const std::string& target_str) const;
};

// ===== Filters =====
// Base filter interface

struct ActivationFilter {
  enum class Target {
    Actor,
    Target
  };
  Target target = Target::Actor;

  virtual ~ActivationFilter() = default;
  virtual bool check(const ActivationContext& ctx) const = 0;
};

// Check if entity has a specific vibe
struct VibeFilter : ActivationFilter {
  ObservationType vibe = 0;
  bool check(const ActivationContext& ctx) const override;
};

// Check if entity has required resources
struct ResourceFilter : ActivationFilter {
  std::string target_str;  // "actor", "target", "actor_commons", "target_commons"
  std::unordered_map<InventoryItem, InventoryQuantity> resources;
  bool check(const ActivationContext& ctx) const override;
};

// Check alignment status of target
struct AlignmentFilter : ActivationFilter {
  std::string alignment;  // "aligned", "unaligned", "same_commons", "different_commons", "not_same_commons"
  bool check(const ActivationContext& ctx) const override;
};

using FilterPtr = std::unique_ptr<ActivationFilter>;

// ===== Mutations =====
// Base mutation interface

struct ActivationMutation {
  virtual ~ActivationMutation() = default;
  virtual bool apply(ActivationContext& ctx) const = 0;
};

// Apply resource deltas to a target
struct ResourceDeltaMutation : ActivationMutation {
  std::string target;  // "actor", "target", "actor_commons", "target_commons"
  std::unordered_map<InventoryItem, InventoryDelta> deltas;
  bool apply(ActivationContext& ctx) const override;
};

// Transfer resources between entities
struct ResourceTransferMutation : ActivationMutation {
  std::string from_target;  // "actor", "target", "actor_commons", "target_commons"
  std::string to_target;
  std::unordered_map<InventoryItem, int> resources;  // -1 = all available
  bool apply(ActivationContext& ctx) const override;
};

// Update commons alignment
struct AlignmentMutation : ActivationMutation {
  std::string align_to;  // "actor_commons" or "none"
  bool apply(ActivationContext& ctx) const override;
};

// Freeze an entity
struct FreezeMutation : ActivationMutation {
  std::string target;  // "actor" or "target"
  int duration = 0;
  bool apply(ActivationContext& ctx) const override;
};

// Clear all resources in a limit group (set to 0)
struct ClearInventoryMutation : ActivationMutation {
  std::string target;      // "actor" or "target"
  std::string limit_name;  // Name of the resource limit group to clear (e.g., "gear")
  bool apply(ActivationContext& ctx) const override;
};

// Forward declaration for recursive type
struct AttackMutation;
using MutationPtr = std::unique_ptr<ActivationMutation>;

// Attack with weapon/armor/defense mechanics
struct AttackMutation : ActivationMutation {
  std::unordered_map<InventoryItem, InventoryQuantity> defense_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> armor_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> weapon_resources;
  std::unordered_map<ObservationType, int> vibe_bonus;
  std::vector<MutationPtr> on_success;

  bool apply(ActivationContext& ctx) const override;

private:
  int compute_weapon_power(const Agent& attacker) const;
  int compute_armor_power(const Agent& target) const;
  bool can_defend(const Agent& target, int damage_bonus) const;
  void consume_defense(Agent& target, int damage_bonus) const;
};

// ===== Activation Handler =====
// Contains filters and mutations to apply on activation

class ActivationHandler {
public:
  std::string name;
  std::vector<FilterPtr> filters;
  std::vector<MutationPtr> mutations;

  ActivationHandler() = default;
  explicit ActivationHandler(const std::string& name) : name(name) {}

  // Check all filters and apply mutations if they pass
  bool try_activate(ActivationContext& ctx) const {
    // Check all filters
    for (const auto& filter : filters) {
      if (!filter->check(ctx)) {
        return false;
      }
    }

    // Apply all mutations
    for (const auto& mutation : mutations) {
      mutation->apply(ctx);
    }

    return true;
  }
};

// Config structures are in activation_handler_config.hpp

// ===== Factory Functions =====

FilterPtr create_filter(const ActivationFilterConfig& config);
MutationPtr create_mutation(const ActivationMutationConfig& config);
std::shared_ptr<ActivationHandler> create_activation_handler(const ActivationHandlerConfig& config);

// Pybind11 bindings are in activation_handler_config.hpp

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTIVATION_HANDLER_HPP_
