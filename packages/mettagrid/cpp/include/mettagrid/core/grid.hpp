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
  GridObject* owner;              // The object that owns this effect
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

class Grid {
public:
  const GridCoord height;
  const GridCoord width;
  vector<std::unique_ptr<GridObject>> objects;

private:
  GridType grid;
  std::vector<std::vector<CellEffects>> _effect_sources;  // Individual effect sources per cell
  std::vector<AOEHelper*> _registered_aoe_helpers;        // All registered AOE helpers
  const std::unordered_map<int, std::string>* _tag_id_map = nullptr;

public:
  Grid(GridCoord height, GridCoord width)
      : height(height),
        width(width),
        objects(),
        grid(),
        _effect_sources(height, std::vector<CellEffects>(width)) {
    grid.resize(height, std::vector<GridObject*>(width, nullptr));

    // Reserve space for objects to avoid frequent reallocations
    size_t estimated_objects = static_cast<size_t>(height) * width / 2;
    constexpr size_t MAX_PREALLOCATED_OBJECTS = 12'500'000;
    size_t reserved_objects = std::min(estimated_objects, MAX_PREALLOCATED_OBJECTS);
    objects.reserve(reserved_objects);

    // GridObjectId "0" is reserved to mean empty space
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
    for (auto& aoe : obj->aoes) {
      aoe.init(this, obj);
      aoe.register_effects(obj->location.r, obj->location.c);
    }

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

  // Remove an object from the grid
  inline bool remove_object(GridObject& obj) {
    if (!is_valid_location(obj.location)) {
      return false;
    }
    if (grid[obj.location.r][obj.location.c] != &obj) {
      return false;
    }

    // Call on_remove for cleanup (includes AOE unregistration)
    obj.on_remove();

    // Clear the grid cell
    grid[obj.location.r][obj.location.c] = nullptr;

    // Release the object
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

  // Get the effect sources at a cell
  const CellEffects& effect_sources_at(GridCoord r, GridCoord c) const {
    return _effect_sources[r][c];
  }

  // Register an AOE effect source at cells within range
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

  // Register an AOE helper for tracking
  void register_aoe_helper(AOEHelper* helper) {
    _registered_aoe_helpers.push_back(helper);
  }

  // Unregister an AOE helper
  void unregister_aoe_helper(AOEHelper* helper) {
    _registered_aoe_helpers.erase(std::remove(_registered_aoe_helpers.begin(), _registered_aoe_helpers.end(), helper),
                                  _registered_aoe_helpers.end());
  }

  // Get all registered AOE helpers
  const std::vector<AOEHelper*>& registered_aoe_helpers() const {
    return _registered_aoe_helpers;
  }

  // Set the tag_id_map for faction tag lookup
  void set_tag_id_map(const std::unordered_map<int, std::string>* tag_id_map) {
    _tag_id_map = tag_id_map;
  }

  // Get the tag_id_map
  const std::unordered_map<int, std::string>* tag_id_map() const {
    return _tag_id_map;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_HPP_
