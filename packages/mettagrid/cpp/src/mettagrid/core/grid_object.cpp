#include "core/grid_object.hpp"

#include <algorithm>

#include "actions/activation_handler.hpp"
#include "config/mettagrid_config.hpp"
#include "core/grid.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/commons.hpp"
#include "objects/has_inventory.hpp"

// GridObject::set_activation_handlers - handlers are applied in registration order
void GridObject::set_activation_handlers(std::vector<std::shared_ptr<ActivationHandler>> handlers) {
  _activation_handlers = std::move(handlers);
}

// GridObject::activate() - try activation handlers
bool GridObject::activate(Agent& actor, Grid* grid, const GameConfig* game_config) {
  if (_activation_handlers.empty()) {
    return false;
  }

  ActivationContext ctx(actor, *this, grid, game_config);

  // Try each handler in registration order
  for (const auto& handler : _activation_handlers) {
    if (handler->try_activate(ctx)) {
      return true;
    }
  }

  return false;
}

// Commons support has been moved to Alignable interface

void AOEHelper::register_effects(GridCoord r, GridCoord c) {
  if (has_aoe()) {
    // Register as individual source for commons-aware filtering
    _grid->register_aoe_source(r, c, _config->range, _owner, _config);
    // Also apply to legacy accumulated effects for backward compatibility
    _grid->apply_aoe(r, c, _config->range, _config->resource_deltas, true);
    _registered = true;
    _location_r = r;
    _location_c = c;

    // Register this AOE helper for object tracking
    _grid->register_aoe_helper(this);

    // Scan for existing HasInventory objects in range and try to register them
    // (skip agents - they are checked every tick since they move)
    int r_start = std::max(0, static_cast<int>(r) - static_cast<int>(_config->range));
    int r_end = std::min(static_cast<int>(_grid->height), static_cast<int>(r) + static_cast<int>(_config->range) + 1);
    int c_start = std::max(0, static_cast<int>(c) - static_cast<int>(_config->range));
    int c_end = std::min(static_cast<int>(_grid->width), static_cast<int>(c) + static_cast<int>(_config->range) + 1);

    for (int row = r_start; row < r_end; ++row) {
      for (int col = c_start; col < c_end; ++col) {
        GridObject* obj = _grid->object_at(GridLocation(row, col));
        if (obj && obj != _owner) {
          try_register_inventory_object(obj);
        }
      }
    }
  }
}

void AOEHelper::unregister_effects() {
  if (_registered && has_aoe()) {
    // Unregister individual source
    _grid->unregister_aoe_source(_location_r, _location_c, _config->range, _owner);
    // Also remove from legacy accumulated effects
    _grid->apply_aoe(_location_r, _location_c, _config->range, _config->resource_deltas, false);

    // Unregister from AOE helper tracking
    _grid->unregister_aoe_helper(this);

    // Clear registered inventory objects
    _registered_inventory_objects.clear();

    _registered = false;
  }
}

bool AOEHelper::try_register_inventory_object(GridObject* obj) {
  // Skip agents - they are checked every tick since they move
  if (dynamic_cast<Agent*>(obj) != nullptr) {
    return false;
  }

  // Check if this object has inventory
  if (dynamic_cast<HasInventory*>(obj) == nullptr) {
    return false;
  }

  // Check if already registered
  if (std::find(_registered_inventory_objects.begin(), _registered_inventory_objects.end(), obj) !=
      _registered_inventory_objects.end()) {
    return true;  // Already registered
  }

  // Check all filters (target_tags + commons)
  const auto* tag_id_map = _grid->tag_id_map();
  if (tag_id_map && !passes_all_filters(obj, *tag_id_map)) {
    return false;
  }

  // All filters passed - register the object
  _registered_inventory_objects.push_back(obj);
  return true;
}

void AOEHelper::unregister_inventory_object(GridObject* obj) {
  _registered_inventory_objects.erase(
      std::remove(_registered_inventory_objects.begin(), _registered_inventory_objects.end(), obj),
      _registered_inventory_objects.end());
}

void AOEHelper::refresh_registrations() {
  if (!_registered || !has_aoe()) {
    return;
  }

  // Clear and re-scan for objects in range
  _registered_inventory_objects.clear();

  int r_start = std::max(0, static_cast<int>(_location_r) - static_cast<int>(_config->range));
  int r_end =
      std::min(static_cast<int>(_grid->height), static_cast<int>(_location_r) + static_cast<int>(_config->range) + 1);
  int c_start = std::max(0, static_cast<int>(_location_c) - static_cast<int>(_config->range));
  int c_end =
      std::min(static_cast<int>(_grid->width), static_cast<int>(_location_c) + static_cast<int>(_config->range) + 1);

  for (int row = r_start; row < r_end; ++row) {
    for (int col = c_start; col < c_end; ++col) {
      GridObject* obj = _grid->object_at(GridLocation(row, col));
      if (obj && obj != _owner) {
        try_register_inventory_object(obj);
      }
    }
  }
}

bool AOEHelper::matches_target_tags(const GridObject* obj) const {
  if (!_config || _config->target_tag_ids.empty()) {
    // No tag filter - match all objects
    return true;
  }

  // Check if any of the object's tags match the target tags
  for (int tag_id : obj->tag_ids) {
    if (std::find(_config->target_tag_ids.begin(), _config->target_tag_ids.end(), tag_id) !=
        _config->target_tag_ids.end()) {
      return true;
    }
  }
  return false;
}

// Helper to get commons name from an object (checks Alignable interface or tags)
static std::string get_commons_name(GridObject* obj, const std::unordered_map<int, std::string>& tag_id_map) {
  const std::string commons_tag_prefix = "commons:";

  // First check if it's an Alignable with a commons
  if (auto* alignable = dynamic_cast<Alignable*>(obj)) {
    if (alignable->getCommons()) {
      return alignable->getCommons()->name;
    }
  }

  // Otherwise check tags for commons
  for (int tag_id : obj->tag_ids) {
    auto tag_it = tag_id_map.find(tag_id);
    if (tag_it != tag_id_map.end()) {
      const std::string& tag_name = tag_it->second;
      if (tag_name.rfind(commons_tag_prefix, 0) == 0) {
        return tag_name.substr(commons_tag_prefix.length());
      }
    }
  }
  return "";
}

// Check if a single AOE filter passes for the given source and target objects
static bool check_aoe_filter(const ActivationFilterConfig& filter,
                             GridObject* source,
                             GridObject* target,
                             const std::unordered_map<int, std::string>& tag_id_map) {
  switch (filter.type) {
    case FilterType::Alignment: {
      // Resolve which object to check based on filter target
      GridObject* check_obj = (filter.alignment.target == TargetType::Actor) ? source : target;

      std::string target_commons = get_commons_name(check_obj, tag_id_map);
      std::string source_commons = get_commons_name(source, tag_id_map);

      switch (filter.alignment.alignment) {
        case AlignmentType::Aligned:
          return !target_commons.empty();
        case AlignmentType::Unaligned:
          return target_commons.empty();
        case AlignmentType::SameCommons:
          return !target_commons.empty() && !source_commons.empty() && target_commons == source_commons;
        case AlignmentType::DifferentCommons:
          // Both must have commons, and they must be different
          return !target_commons.empty() && !source_commons.empty() && target_commons != source_commons;
        case AlignmentType::NotSameCommons:
          // NOT aligned to source: either unaligned or aligned to different commons
          return target_commons.empty() || target_commons != source_commons;
      }
      break;
    }
    case FilterType::Vibe: {
      // Check vibe of target or source
      GridObject* check_obj = (filter.vibe.target == TargetType::Actor) ? source : target;
      return check_obj->vibe == filter.vibe.vibe;
    }
    case FilterType::Resource: {
      // Check resources of target or source
      GridObject* check_obj = (filter.resource.target == TargetType::Actor) ? source : target;
      auto* has_inv = dynamic_cast<HasInventory*>(check_obj);
      if (!has_inv) return false;
      for (const auto& [item, amount] : filter.resource.resources) {
        if (has_inv->inventory.amount(item) < amount) return false;
      }
      break;
    }
  }
  return true;
}

bool AOEHelper::passes_all_filters(GridObject* obj, const std::unordered_map<int, std::string>& tag_id_map) const {
  if (!_config) {
    return false;
  }

  // Check target_tags filter first
  if (!matches_target_tags(obj)) {
    return false;
  }

  // Check all configured filters
  for (const auto& filter : _config->filters) {
    if (!check_aoe_filter(filter, _owner, obj, tag_id_map)) {
      return false;
    }
  }

  return true;
}

// Grid notification methods
void Grid::notify_object_added(GridObject* obj) {
  // Check all registered AOE helpers to see if this object should be registered
  for (AOEHelper* helper : _registered_aoe_helpers) {
    if (!helper->is_registered() || !helper->config()) {
      continue;
    }

    // Check if object is in range of this AOE
    if (is_in_range(
            obj->location.r, obj->location.c, helper->location_r(), helper->location_c(), helper->config()->range)) {
      helper->try_register_inventory_object(obj);
    }
  }
}

void Grid::notify_object_removed(GridObject* obj) {
  // Notify all AOE helpers to unregister this object
  for (AOEHelper* helper : _registered_aoe_helpers) {
    helper->unregister_inventory_object(obj);
  }
}

void Grid::refresh_aoe_registrations() {
  for (AOEHelper* helper : _registered_aoe_helpers) {
    helper->refresh_registrations();
  }
}
