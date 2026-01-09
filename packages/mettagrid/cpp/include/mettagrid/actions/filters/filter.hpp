#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_FILTERS_FILTER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_FILTERS_FILTER_HPP_

#include "actions/activation_context.hpp"
#include "actions/activation_handler_config.hpp"
#include "core/grid_object.hpp"

namespace mettagrid {

/**
 * Base interface for activation filters.
 * Filters determine whether an activation should proceed.
 */
class Filter {
public:
  virtual ~Filter() = default;

  // Returns true if the activation passes this filter
  virtual bool passes(const ActivationContext& ctx) const = 0;
};

// ============================================================================
// Filter Implementations
// ============================================================================

/**
 * VibeFilter: Check if entity has a specific vibe
 */
class VibeFilter : public Filter {
public:
  explicit VibeFilter(const VibeFilterConfig& config) : _config(config) {}

  bool passes(const ActivationContext& ctx) const override {
    // Cast to GridObject to access vibe (GridObject inherits from HasVibe)
    GridObject* grid_obj = dynamic_cast<GridObject*>(ctx.resolve(_config.entity));
    if (grid_obj == nullptr) {
      return false;
    }

    return grid_obj->vibe == _config.vibe_id;
  }

private:
  VibeFilterConfig _config;
};

/**
 * ResourceFilter: Check if entity has minimum resources
 */
class ResourceFilter : public Filter {
public:
  explicit ResourceFilter(const ResourceFilterConfig& config) : _config(config) {}

  bool passes(const ActivationContext& ctx) const override {
    HasInventory* entity = ctx.resolve(_config.entity);
    if (entity == nullptr) {
      return false;
    }

    return entity->inventory.amount(_config.resource_id) >= _config.min_amount;
  }

private:
  ResourceFilterConfig _config;
};

/**
 * AlignmentFilter: Check alignment relationships
 */
class AlignmentFilter : public Filter {
public:
  explicit AlignmentFilter(const AlignmentFilterConfig& config) : _config(config) {}

  bool passes(const ActivationContext& ctx) const override {
    Collective* actor_coll = ctx.actor_collective();
    Collective* target_coll = ctx.target_collective();

    switch (_config.condition) {
      case AlignmentCondition::aligned:
        return actor_coll != nullptr && target_coll != nullptr;

      case AlignmentCondition::unaligned:
        return actor_coll == nullptr || target_coll == nullptr;

      case AlignmentCondition::same_collective:
        return actor_coll != nullptr && actor_coll == target_coll;

      case AlignmentCondition::different_collective:
        return actor_coll != nullptr && target_coll != nullptr && actor_coll != target_coll;

      default:
        return false;
    }
  }

private:
  AlignmentFilterConfig _config;
};

/**
 * TagFilter: Check if entity has required tags
 */
class TagFilter : public Filter {
public:
  explicit TagFilter(const TagFilterConfig& config) : _config(config) {}

  bool passes(const ActivationContext& ctx) const override {
    // Get GridObject to access tag_ids
    GridObject* grid_obj = dynamic_cast<GridObject*>(ctx.resolve(_config.entity));
    if (grid_obj == nullptr) {
      return false;
    }

    // Empty required tags means pass
    if (_config.required_tag_ids.empty()) {
      return true;
    }

    // Check if entity has any of the required tags
    for (int required_tag : _config.required_tag_ids) {
      for (int entity_tag : grid_obj->tag_ids) {
        if (required_tag == entity_tag) {
          return true;
        }
      }
    }

    return false;
  }

private:
  TagFilterConfig _config;
};

}  // namespace mettagrid

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_FILTERS_FILTER_HPP_
