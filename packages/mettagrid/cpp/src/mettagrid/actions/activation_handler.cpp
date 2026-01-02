#include "actions/activation_handler.hpp"

#include <algorithm>

#include "config/mettagrid_config.hpp"
#include "core/grid.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/commons.hpp"
#include "objects/has_inventory.hpp"

// ===== ActivationContext implementations =====

Inventory* ActivationContext::resolve_inventory(const std::string& target_str) const {
  if (target_str == "actor") {
    return &actor.inventory;
  } else if (target_str == "target") {
    HasInventory* has_inv = dynamic_cast<HasInventory*>(&target);
    if (has_inv) return &has_inv->inventory;
    return nullptr;
  } else if (target_str == "actor_commons") {
    Commons* commons = actor.getCommons();
    if (commons) return &commons->inventory;
    return nullptr;
  } else if (target_str == "target_commons") {
    Alignable* alignable = dynamic_cast<Alignable*>(&target);
    if (alignable) {
      Commons* commons = alignable->getCommons();
      if (commons) return &commons->inventory;
    }
    return nullptr;
  }
  return nullptr;
}

Alignable* ActivationContext::resolve_alignable(const std::string& target_str) const {
  if (target_str == "target") {
    return dynamic_cast<Alignable*>(&target);
  } else if (target_str == "actor") {
    return dynamic_cast<Alignable*>(&actor);
  }
  return nullptr;
}

Commons* ActivationContext::resolve_commons(const std::string& target_str) const {
  if (target_str == "actor_commons") {
    return actor.getCommons();
  } else if (target_str == "target_commons") {
    Alignable* alignable = dynamic_cast<Alignable*>(&target);
    if (alignable) return alignable->getCommons();
    return nullptr;
  }
  return nullptr;
}

Agent* ActivationContext::resolve_agent(const std::string& target_str) const {
  if (target_str == "actor") {
    return &actor;
  } else if (target_str == "target") {
    return dynamic_cast<Agent*>(&target);
  }
  return nullptr;
}

// ===== Filter implementations =====

bool VibeFilter::check(const ActivationContext& ctx) const {
  if (target == Target::Actor) {
    return ctx.actor.vibe == vibe;
  }
  // Target could be any GridObject with vibe
  return ctx.target.vibe == vibe;
}

bool ResourceFilter::check(const ActivationContext& ctx) const {
  Inventory* inv = ctx.resolve_inventory(target_str);
  if (!inv) return false;
  for (const auto& [item, amount] : resources) {
    if (inv->amount(item) < amount) return false;
  }
  return true;
}

bool AlignmentFilter::check(const ActivationContext& ctx) const {
  // Resolve the alignable target
  Alignable* alignable = ctx.resolve_alignable(target == Target::Actor ? "actor" : "target");
  if (!alignable) return false;

  Commons* target_commons = alignable->getCommons();
  Commons* actor_commons = ctx.actor.getCommons();

  if (alignment == "aligned") {
    return target_commons != nullptr;
  } else if (alignment == "unaligned") {
    return target_commons == nullptr;
  } else if (alignment == "same_commons") {
    return target_commons != nullptr && target_commons == actor_commons;
  } else if (alignment == "different_commons") {
    return target_commons != nullptr && target_commons != actor_commons;
  } else if (alignment == "not_same_commons") {
    // NOT aligned to actor: either unaligned or aligned to different commons
    return target_commons == nullptr || target_commons != actor_commons;
  }
  return false;
}

// ===== Mutation implementations =====

bool ResourceDeltaMutation::apply(ActivationContext& ctx) const {
  Inventory* inv = ctx.resolve_inventory(target);
  if (!inv) return false;
  for (const auto& [item, delta] : deltas) {
    inv->update(item, delta);
  }
  return true;
}

bool ResourceTransferMutation::apply(ActivationContext& ctx) const {
  Inventory* from_inv = ctx.resolve_inventory(from_target);
  Inventory* to_inv = ctx.resolve_inventory(to_target);
  if (!from_inv || !to_inv) return false;

  for (const auto& [item, amount] : resources) {
    int transfer_amount = amount;
    if (transfer_amount == -1) {
      // Transfer all available
      transfer_amount = from_inv->amount(item);
    }
    if (transfer_amount > 0) {
      InventoryDelta actual = to_inv->update(item, transfer_amount);
      if (actual > 0) {
        from_inv->update(item, -actual);
      }
    }
  }
  return true;
}

bool AlignmentMutation::apply(ActivationContext& ctx) const {
  Alignable* alignable = ctx.resolve_alignable("target");
  if (!alignable) return false;

  Commons* target_commons = alignable->getCommons();

  if (align_to == "none") {
    // Scramble: target must have a commons to clear
    if (target_commons == nullptr) return false;
    alignable->setCommons(nullptr);
    return true;
  } else if (align_to == "actor_commons") {
    // Align: actor must have commons, target must be unaligned
    Commons* actor_commons = ctx.resolve_commons("actor_commons");
    if (actor_commons == nullptr || target_commons != nullptr) return false;
    alignable->setCommons(actor_commons);
    return true;
  }
  return false;
}

bool FreezeMutation::apply(ActivationContext& ctx) const {
  Agent* agent = ctx.resolve_agent(target);
  if (!agent) return false;
  agent->frozen = duration;
  return true;
}

bool ClearInventoryMutation::apply(ActivationContext& ctx) const {
  Inventory* inv = ctx.resolve_inventory(target);
  if (!inv || !ctx.game_config) return false;

  // Look up resources for this limit name
  auto it = ctx.game_config->inventory_limit_resources.find(limit_name);
  if (it == ctx.game_config->inventory_limit_resources.end()) return false;

  for (const auto& item : it->second) {
    InventoryQuantity current = inv->amount(item);
    if (current > 0) {
      inv->update(item, -static_cast<InventoryDelta>(current));
    }
  }
  return true;
}

// ===== AttackMutation implementations =====

bool AttackMutation::apply(ActivationContext& ctx) const {
  Agent* target_agent = ctx.resolve_agent("target");
  if (!target_agent) return false;

  // Don't attack frozen agents (allow swap instead)
  if (target_agent->frozen > 0) return false;

  int weapon_power = compute_weapon_power(ctx.actor);
  int armor_power = compute_armor_power(*target_agent);
  int damage_bonus = std::max(weapon_power - armor_power, 0);

  // Check if target can defend
  if (can_defend(*target_agent, damage_bonus)) {
    consume_defense(*target_agent, damage_bonus);
    // Attack blocked - still counts as successful activation
    ctx.actor.stats.incr("action.attack." + ctx.actor.group_name + ".blocked_by." + target_agent->group_name);
    return true;
  }

  // Attack lands - apply on_success mutations
  for (const auto& mutation : on_success) {
    mutation->apply(ctx);
  }

  // Log successful attack
  bool same_team = (ctx.actor.group_name == target_agent->group_name);
  if (same_team) {
    ctx.actor.stats.incr("action.attack." + ctx.actor.group_name + ".friendly_fire");
  } else {
    ctx.actor.stats.incr("action.attack." + ctx.actor.group_name + ".hit." + target_agent->group_name);
    target_agent->stats.incr("action.attack." + target_agent->group_name + ".hit_by." + ctx.actor.group_name);
  }

  return true;
}

int AttackMutation::compute_weapon_power(const Agent& attacker) const {
  int power = 0;
  for (const auto& [item, weight] : weapon_resources) {
    power += attacker.inventory.amount(item) * weight;
  }
  return power;
}

int AttackMutation::compute_armor_power(const Agent& target) const {
  int power = 0;
  for (const auto& [item, weight] : armor_resources) {
    int amount = target.inventory.amount(item);
    // Check if target is vibing a resource that matches armor
    auto vibe_it = vibe_bonus.find(target.vibe);
    if (vibe_it != vibe_bonus.end()) {
      // Add vibe bonus if target is vibing
      amount += vibe_it->second;
    }
    power += amount * weight;
  }
  return power;
}

bool AttackMutation::can_defend(const Agent& target, int damage_bonus) const {
  if (defense_resources.empty()) return false;
  for (const auto& [item, amount] : defense_resources) {
    int required = static_cast<int>(amount) + damage_bonus;
    if (target.inventory.amount(item) < static_cast<InventoryQuantity>(required)) {
      return false;
    }
  }
  return true;
}

void AttackMutation::consume_defense(Agent& target, int damage_bonus) const {
  for (const auto& [item, amount] : defense_resources) {
    int required = static_cast<int>(amount) + damage_bonus;
    target.inventory.update(item, -required);
  }
}

// ===== Factory function implementations =====

FilterPtr create_filter(const ActivationFilterConfig& config) {
  if (config.type == "vibe") {
    auto filter = std::make_unique<VibeFilter>();
    filter->target =
        config.vibe.target == "target" ? ActivationFilter::Target::Target : ActivationFilter::Target::Actor;
    filter->vibe = config.vibe.vibe;
    return filter;
  } else if (config.type == "resource") {
    auto filter = std::make_unique<ResourceFilter>();
    filter->target =
        config.resource.target == "target" ? ActivationFilter::Target::Target : ActivationFilter::Target::Actor;
    filter->target_str = config.resource.target;  // Store full target string for commons support
    filter->resources = config.resource.resources;
    return filter;
  } else if (config.type == "alignment") {
    auto filter = std::make_unique<AlignmentFilter>();
    filter->target =
        config.alignment.target == "target" ? ActivationFilter::Target::Target : ActivationFilter::Target::Actor;
    filter->alignment = config.alignment.alignment;
    return filter;
  }
  return nullptr;
}

MutationPtr create_mutation(const ActivationMutationConfig& config) {
  if (config.type == "resource_delta") {
    auto mutation = std::make_unique<ResourceDeltaMutation>();
    mutation->target = config.resource_delta.target;
    mutation->deltas = config.resource_delta.deltas;
    return mutation;
  } else if (config.type == "resource_transfer") {
    auto mutation = std::make_unique<ResourceTransferMutation>();
    mutation->from_target = config.resource_transfer.from_target;
    mutation->to_target = config.resource_transfer.to_target;
    mutation->resources = config.resource_transfer.resources;
    return mutation;
  } else if (config.type == "alignment") {
    auto mutation = std::make_unique<AlignmentMutation>();
    mutation->align_to = config.alignment.align_to;
    return mutation;
  } else if (config.type == "freeze") {
    auto mutation = std::make_unique<FreezeMutation>();
    mutation->target = config.freeze.target;
    mutation->duration = config.freeze.duration;
    return mutation;
  } else if (config.type == "clear_inventory") {
    auto mutation = std::make_unique<ClearInventoryMutation>();
    mutation->target = config.clear_inventory.target;
    mutation->limit_name = config.clear_inventory.limit_name;
    return mutation;
  } else if (config.type == "attack") {
    auto mutation = std::make_unique<AttackMutation>();
    mutation->defense_resources = config.attack.defense_resources;
    mutation->armor_resources = config.attack.armor_resources;
    mutation->weapon_resources = config.attack.weapon_resources;
    mutation->vibe_bonus = config.attack.vibe_bonus;
    for (const auto& on_success_cfg : config.attack.on_success) {
      if (on_success_cfg) {
        mutation->on_success.push_back(create_mutation(*on_success_cfg));
      }
    }
    return mutation;
  }
  return nullptr;
}

std::shared_ptr<ActivationHandler> create_activation_handler(const ActivationHandlerConfig& config) {
  auto handler = std::make_shared<ActivationHandler>(config.name);
  for (const auto& filter_cfg : config.filters) {
    auto filter = create_filter(filter_cfg);
    if (filter) {
      handler->filters.push_back(std::move(filter));
    }
  }
  for (const auto& mutation_cfg : config.mutations) {
    auto mutation = create_mutation(mutation_cfg);
    if (mutation) {
      handler->mutations.push_back(std::move(mutation));
    }
  }
  return handler;
}
