#include "core/grid_object.hpp"

#include <algorithm>

#include "core/grid.hpp"
#include "objects/agent.hpp"
#include "objects/alignable.hpp"
#include "objects/commons.hpp"
#include "objects/has_inventory.hpp"

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

bool AOEHelper::passes_all_filters(GridObject* obj, const std::unordered_map<int, std::string>& tag_id_map) const {
  if (!_config) {
    return false;
  }

  // Check target_tags filter first
  if (!matches_target_tags(obj)) {
    return false;
  }

  // Check commons membership filtering
  const std::string commons_tag_prefix = "commons:";

  // Determine if source and target share the same commons
  bool same_commons = false;

  // Get source's commons
  std::string source_commons_name;
  if (auto* source_alignable = dynamic_cast<Alignable*>(_owner)) {
    if (source_alignable->getCommons()) {
      source_commons_name = source_alignable->getCommons()->name;
    }
  } else if (_owner) {
    // Source is not Alignable, check tags for commons
    for (int tag_id : _owner->tag_ids) {
      auto tag_it = tag_id_map.find(tag_id);
      if (tag_it != tag_id_map.end()) {
        const std::string& tag_name = tag_it->second;
        if (tag_name.rfind(commons_tag_prefix, 0) == 0) {
          source_commons_name = tag_name.substr(commons_tag_prefix.length());
          break;
        }
      }
    }
  }

  // Get target's commons and compare
  if (auto* target_alignable = dynamic_cast<Alignable*>(obj)) {
    if (target_alignable->getCommons() && !source_commons_name.empty()) {
      same_commons = (target_alignable->getCommons()->name == source_commons_name);
    }
  } else {
    // Target is not Alignable, check tags for commons
    for (int tag_id : obj->tag_ids) {
      auto tag_it = tag_id_map.find(tag_id);
      if (tag_it != tag_id_map.end()) {
        const std::string& tag_name = tag_it->second;
        if (tag_name.rfind(commons_tag_prefix, 0) == 0) {
          std::string target_commons_name = tag_name.substr(commons_tag_prefix.length());
          if (target_commons_name == source_commons_name) {
            same_commons = true;
            break;
          }
        }
      }
    }
  }

  // Apply members_only and ignore_members filters
  if (_config->members_only && !same_commons) {
    return false;  // Effect only for members, but target is not a member
  }
  if (_config->ignore_members) {
    // If source has no commons, treat everyone as a member (skip all)
    if (source_commons_name.empty() || same_commons) {
      return false;  // Effect ignores members, and target is a member (or source has no alignment)
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
