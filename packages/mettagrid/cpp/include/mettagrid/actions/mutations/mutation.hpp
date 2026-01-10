#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MUTATIONS_MUTATION_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MUTATIONS_MUTATION_HPP_

#include <algorithm>
#include <cmath>

#include "actions/activation_context.hpp"
#include "actions/activation_handler_config.hpp"
#include "objects/has_inventory.hpp"

namespace mettagrid {

/**
 * Base interface for activation mutations.
 * Mutations modify state when an activation occurs.
 */
class Mutation {
public:
  virtual ~Mutation() = default;

  // Apply the mutation to the context
  virtual void apply(ActivationContext& ctx) = 0;
};

// ============================================================================
// Mutation Implementations
// ============================================================================

/**
 * ResourceDeltaMutation: Add/remove resources from an entity
 */
class ResourceDeltaMutation : public Mutation {
public:
  explicit ResourceDeltaMutation(const ResourceDeltaMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    HasInventory* entity = ctx.resolve(_config.entity);
    if (entity == nullptr) {
      return;
    }

    entity->inventory.update(_config.resource_id, _config.delta);
  }

private:
  ResourceDeltaMutationConfig _config;
};

/**
 * ResourceTransferMutation: Move resources between entities
 */
class ResourceTransferMutation : public Mutation {
public:
  explicit ResourceTransferMutation(const ResourceTransferMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    HasInventory* source = ctx.resolve(_config.source);
    HasInventory* dest = ctx.resolve(_config.destination);

    if (source == nullptr || dest == nullptr) {
      return;
    }

    InventoryDelta amount = _config.amount;
    if (amount < 0) {
      // Transfer all available
      amount = static_cast<InventoryDelta>(source->inventory.amount(_config.resource_id));
    }

    HasInventory::transfer_resources(source->inventory,
                                     dest->inventory,
                                     _config.resource_id,
                                     amount,
                                     false  // Don't destroy untransferred resources
    );
  }

private:
  ResourceTransferMutationConfig _config;
};

/**
 * AlignmentMutation: Change target's collective alignment
 */
class AlignmentMutation : public Mutation {
public:
  explicit AlignmentMutation(const AlignmentMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    // All GridObjects are Alignable - try to cast target to GridObject
    GridObject* target_obj = dynamic_cast<GridObject*>(ctx.target);
    if (target_obj == nullptr) {
      fprintf(stderr,
              "AlignmentMutation::apply - FAILED: dynamic_cast returned null for ctx.target=%p\n",
              static_cast<void*>(ctx.target));
      return;
    }

    switch (_config.align_to) {
      case AlignTo::actor_collective: {
        Collective* actor_coll = ctx.actor_collective();
        fprintf(stderr,
                "AlignmentMutation::apply - Align to actor_collective (actor_coll=%p) on %s\n",
                static_cast<void*>(actor_coll),
                target_obj->type_name.c_str());
        if (actor_coll != nullptr) {
          target_obj->setCollective(actor_coll);
        }
        break;
      }
      case AlignTo::none:
        fprintf(stderr,
                "AlignmentMutation::apply - RemoveAlignment on %s (before: collective=%p)\n",
                target_obj->type_name.c_str(),
                static_cast<void*>(target_obj->getCollective()));
        target_obj->clearCollective();
        fprintf(stderr,
                "AlignmentMutation::apply - RemoveAlignment complete (after: collective=%p)\n",
                static_cast<void*>(target_obj->getCollective()));
        break;
    }
  }

private:
  AlignmentMutationConfig _config;
};

/**
 * FreezeMutation: Freeze target for duration
 * Note: This requires the target to have a freeze mechanism.
 * For now, this is a stub that would need integration with agent freeze system.
 */
class FreezeMutation : public Mutation {
public:
  explicit FreezeMutation(const FreezeMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    // TODO: Integrate with agent freeze system
    // For now, this is a placeholder
    (void)ctx;
    (void)_config;
  }

private:
  FreezeMutationConfig _config;
};

/**
 * ClearInventoryMutation: Clear resources from entity
 */
class ClearInventoryMutation : public Mutation {
public:
  explicit ClearInventoryMutation(const ClearInventoryMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    HasInventory* entity = ctx.resolve(_config.entity);
    if (entity == nullptr) {
      return;
    }

    if (_config.resource_ids.empty()) {
      // Clear all resources
      auto items = entity->inventory.get();
      for (const auto& [item, amount] : items) {
        entity->inventory.update(item, -static_cast<InventoryDelta>(amount));
      }
    } else {
      // Clear specific resources in the list
      for (const auto& resource_id : _config.resource_ids) {
        InventoryQuantity amount = entity->inventory.amount(resource_id);
        entity->inventory.update(resource_id, -static_cast<InventoryDelta>(amount));
      }
    }
  }

private:
  ClearInventoryMutationConfig _config;
};

/**
 * AttackMutation: Combat with weapon/armor/health
 */
class AttackMutation : public Mutation {
public:
  explicit AttackMutation(const AttackMutationConfig& config) : _config(config) {}

  void apply(ActivationContext& ctx) override {
    if (ctx.actor == nullptr || ctx.target == nullptr) {
      return;
    }

    // Get weapon power from actor
    InventoryQuantity weapon = ctx.actor->inventory.amount(_config.weapon_resource);

    // Get armor from target
    InventoryQuantity armor = ctx.target->inventory.amount(_config.armor_resource);

    // Calculate damage
    float raw_damage = static_cast<float>(weapon) * _config.damage_multiplier;
    float damage = std::max(0.0f, raw_damage - static_cast<float>(armor));

    // Apply damage to target's health
    if (damage > 0) {
      ctx.target->inventory.update(_config.health_resource, -static_cast<InventoryDelta>(damage));
    }
  }

private:
  AttackMutationConfig _config;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MUTATIONS_MUTATION_HPP_
