#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "objects/constants.hpp"

using std::max;
using std::unique_ptr;
using std::vector;
using GridType = std::vector<std::vector<GridObject*>>;

// A single AOE effect source at a cell
struct AOEEffectSource {
  GridObject* owner;              // The object that owns this effect (for commons checking)
  const AOEEffectConfig* config;  // Pointer to the effect configuration

  AOEEffectSource(GridObject* owner, const AOEEffectConfig* config) : owner(owner), config(config) {}
};

// Collection of AOE effect sources at a single grid cell
struct CellEffects {
  std::vector<AOEEffectSource> sources;

  void add_source(GridObject* owner, const AOEEffectConfig* config) {
    sources.emplace_back(owner, config);
  }

  void remove_source(GridObject* owner) {
    sources.erase(
        std::remove_if(sources.begin(), sources.end(), [owner](const AOEEffectSource& s) { return s.owner == owner; }),
        sources.end());
  }

  bool empty() const {
    return sources.empty();
  }
};

// Legacy CellEffect for backward compatibility - accumulated resource effects
struct CellEffect {
  std::unordered_map<InventoryItem, InventoryDelta> resource_deltas;

  void add(InventoryItem item, InventoryDelta delta) {
    resource_deltas[item] += delta;
    if (resource_deltas[item] == 0) {
      resource_deltas.erase(item);
    }
  }

  void subtract(InventoryItem item, InventoryDelta delta) {
    add(item, -delta);
  }

  bool empty() const {
    return resource_deltas.empty();
  }
};

class Grid {
public:
  const GridCoord height;
  const GridCoord width;
  vector<std::unique_ptr<GridObject>> objects;

private:
  GridType grid;
  std::vector<std::vector<CellEffect>> _effects;                      // Legacy: accumulated effects (for simple AOE)
  std::vector<std::vector<CellEffects>> _effect_sources;              // New: individual effect sources per cell
  std::vector<AOEHelper*> _registered_aoe_helpers;                    // All registered AOE helpers for object tracking
  const std::unordered_map<int, std::string>* _tag_id_map = nullptr;  // For commons tag lookup

public:
  Grid(GridCoord height, GridCoord width)
      : height(height),
        width(width),
        objects(),
        grid(),
        _effects(height, std::vector<CellEffect>(width)),
        _effect_sources(height, std::vector<CellEffects>(width)) {
    grid.resize(height, std::vector<GridObject*>(width, nullptr));

    // Reserve space for objects to avoid frequent reallocations
    size_t estimated_objects = static_cast<size_t>(height) * width / 2;
    constexpr size_t MAX_PREALLOCATED_OBJECTS = 12'500'000;
    size_t reserved_objects = std::min(estimated_objects, MAX_PREALLOCATED_OBJECTS);
    objects.reserve(reserved_objects);

    // GridObjectId "0" is reserved to mean empty space (GridObject pointer = nullptr).
    objects.push_back(nullptr);
  }

  ~Grid() = default;

  inline bool is_valid_location(const GridLocation& loc) const {
    return loc.r < height && loc.c < width;
  }

  inline bool add_object(GridObject* obj) {
    if (!is_valid_location(obj->location)) {
      return false;
    }
    if (this->grid[obj->location.r][obj->location.c] != nullptr) {
      return false;
    }

    obj->id = static_cast<GridObjectId>(this->objects.size());
    this->objects.push_back(std::unique_ptr<GridObject>(obj));
    this->grid[obj->location.r][obj->location.c] = obj;

    // Register AOE effects if configured
    obj->aoe.init(this, obj);  // Pass owner for commons checking
    obj->aoe.register_effects(obj->location.r, obj->location.c);

    // Notify existing AOE sources about this new object
    notify_object_added(obj);

    return true;
  }

  inline bool move_object(GridObject& obj, const GridLocation& loc) {
    if (!is_valid_location(loc)) {
      return false;
    }

    if (grid[loc.r][loc.c] != nullptr) {
      return false;
    }

    grid[loc.r][loc.c] = &obj;
    grid[obj.location.r][obj.location.c] = nullptr;
    obj.location = loc;
    return true;
  }

  inline bool swap_objects(GridObject& obj1, GridObject& obj2) {
    GridLocation loc1 = obj1.location;
    GridLocation loc2 = obj2.location;

    grid[loc1.r][loc1.c] = &obj2;
    grid[loc2.r][loc2.c] = &obj1;
    obj1.location = loc2;
    obj2.location = loc1;
    return true;
  }

  // Remove an object from the grid. Returns true if successful.
  // Note: The object's ID slot in the objects vector becomes null but is not reused,
  // keeping all other IDs stable.
  inline bool remove_object(GridObject& obj) {
    if (!is_valid_location(obj.location)) {
      return false;
    }
    if (grid[obj.location.r][obj.location.c] != &obj) {
      return false;  // Object not at expected location
    }

    // Notify AOE sources before removing (in case this object was registered with them)
    notify_object_removed(&obj);

    // Call on_remove for cleanup (includes AOE unregistration)
    obj.on_remove();

    // Clear the grid cell
    grid[obj.location.r][obj.location.c] = nullptr;

    // Release the object (unique_ptr becomes null but slot remains)
    if (obj.id < objects.size()) {
      objects[obj.id].reset();
    }
    return true;
  }

  inline GridObject* object(GridObjectId obj_id) const {
    assert(obj_id < objects.size() && "Invalid object ID");
    return objects[obj_id].get();
  }

  inline GridObject* object_at(const GridLocation& loc) const {
    if (!is_valid_location(loc)) {
      return nullptr;
    }
    return grid[loc.r][loc.c];
  }

  inline bool is_empty(GridCoord row, GridCoord col) const {
    return grid[row][col] == nullptr;
  }

  // Get the legacy accumulated effect at a cell (for simple AOE without commons filtering)
  const CellEffect& effect_at(GridCoord r, GridCoord c) const {
    return _effects[r][c];
  }

  // Get the individual effect sources at a cell (for AOE with commons filtering)
  const CellEffects& effect_sources_at(GridCoord r, GridCoord c) const {
    return _effect_sources[r][c];
  }

  // Register an individual AOE effect source at cells within range
  void register_aoe_source(GridCoord center_r,
                           GridCoord center_c,
                           unsigned int radius,
                           GridObject* owner,
                           const AOEEffectConfig* config) {
    int r_start = std::max(0, static_cast<int>(center_r) - static_cast<int>(radius));
    int r_end = std::min(static_cast<int>(height), static_cast<int>(center_r) + static_cast<int>(radius) + 1);
    int c_start = std::max(0, static_cast<int>(center_c) - static_cast<int>(radius));
    int c_end = std::min(static_cast<int>(width), static_cast<int>(center_c) + static_cast<int>(radius) + 1);

    for (int r = r_start; r < r_end; ++r) {
      for (int c = c_start; c < c_end; ++c) {
        _effect_sources[r][c].add_source(owner, config);
      }
    }
  }

  // Unregister an AOE effect source from cells within range
  void unregister_aoe_source(GridCoord center_r, GridCoord center_c, unsigned int radius, GridObject* owner) {
    int r_start = std::max(0, static_cast<int>(center_r) - static_cast<int>(radius));
    int r_end = std::min(static_cast<int>(height), static_cast<int>(center_r) + static_cast<int>(radius) + 1);
    int c_start = std::max(0, static_cast<int>(center_c) - static_cast<int>(radius));
    int c_end = std::min(static_cast<int>(width), static_cast<int>(center_c) + static_cast<int>(radius) + 1);

    for (int r = r_start; r < r_end; ++r) {
      for (int c = c_start; c < c_end; ++c) {
        _effect_sources[r][c].remove_source(owner);
      }
    }
  }

  // Apply AOE effects from a source at (center_r, center_c) with given radius (legacy method)
  // If adding=true, adds effects; if false, removes them
  void apply_aoe(GridCoord center_r,
                 GridCoord center_c,
                 unsigned int radius,
                 const std::unordered_map<InventoryItem, InventoryDelta>& resource_deltas,
                 bool adding) {
    int r_start = std::max(0, static_cast<int>(center_r) - static_cast<int>(radius));
    int r_end = std::min(static_cast<int>(height), static_cast<int>(center_r) + static_cast<int>(radius) + 1);
    int c_start = std::max(0, static_cast<int>(center_c) - static_cast<int>(radius));
    int c_end = std::min(static_cast<int>(width), static_cast<int>(center_c) + static_cast<int>(radius) + 1);

    for (int r = r_start; r < r_end; ++r) {
      for (int c = c_start; c < c_end; ++c) {
        for (const auto& [item, delta] : resource_deltas) {
          if (adding) {
            _effects[r][c].add(item, delta);
          } else {
            _effects[r][c].subtract(item, delta);
          }
        }
      }
    }
  }

  // Register an AOE helper for tracking (called when AOE is registered)
  void register_aoe_helper(AOEHelper* helper) {
    _registered_aoe_helpers.push_back(helper);
  }

  // Unregister an AOE helper (called when AOE is unregistered)
  void unregister_aoe_helper(AOEHelper* helper) {
    _registered_aoe_helpers.erase(std::remove(_registered_aoe_helpers.begin(), _registered_aoe_helpers.end(), helper),
                                  _registered_aoe_helpers.end());
  }

  // Check if a location is within range of an AOE center (Chebyshev/square distance)
  static bool is_in_range(GridCoord obj_r,
                          GridCoord obj_c,
                          GridCoord center_r,
                          GridCoord center_c,
                          unsigned int range) {
    int dr = std::abs(static_cast<int>(obj_r) - static_cast<int>(center_r));
    int dc = std::abs(static_cast<int>(obj_c) - static_cast<int>(center_c));
    return static_cast<unsigned int>(std::max(dr, dc)) <= range;
  }

  // Get all registered AOE helpers
  const std::vector<AOEHelper*>& registered_aoe_helpers() const {
    return _registered_aoe_helpers;
  }

  // Set the tag_id_map for commons tag lookup (call from MettaGrid constructor)
  void set_tag_id_map(const std::unordered_map<int, std::string>* tag_id_map) {
    _tag_id_map = tag_id_map;
  }

  // Get the tag_id_map
  const std::unordered_map<int, std::string>* tag_id_map() const {
    return _tag_id_map;
  }

  // Notify all AOE helpers about a new object being added
  // Call this after adding an object that might be affected by AOEs
  void notify_object_added(GridObject* obj);

  // Notify all AOE helpers about an object being removed
  // Call this before removing an object
  void notify_object_removed(GridObject* obj);

  // Refresh all AOE registrations (call when alignment changes)
  void refresh_aoe_registrations();
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
