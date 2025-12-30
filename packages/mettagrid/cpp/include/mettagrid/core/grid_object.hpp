#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/has_vibe.hpp"

// Forward declarations for AOEHelper
class Grid;
class GridObject;

using TypeId = ObservationType;
using ObservationCoord = ObservationType;
using Vibe = ObservationType;

struct PartialObservationToken {
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
};

// These may make more sense in observation_encoder.hpp, but we need to include that
// header in a lot of places, and it's nice to have these types defined in one place.
struct alignas(1) ObservationToken {
  ObservationType location = EmptyTokenByte;
  ObservationType feature_id = EmptyTokenByte;
  ObservationType value = EmptyTokenByte;
};

// The alignas should make sure of this, but let's be explicit.
// We're going to be reinterpret_casting things to this type, so
// it'll be bad if the compiler pads this type.
static_assert(sizeof(ObservationToken) == 3 * sizeof(uint8_t), "ObservationToken must be 3 bytes");

using ObservationTokens = std::span<ObservationToken>;

class GridLocation {
public:
  GridCoord r;
  GridCoord c;

  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c) {}
  inline GridLocation() : r(0), c(0) {}

  inline bool operator==(const GridLocation& other) const {
    return r == other.r && c == other.c;
  }
};

// Configuration for Area of Effect (AOE) resource effects
struct AOEEffectConfig {
  unsigned int range = 1;                                             // Radius of effect (Manhattan distance)
  std::unordered_map<InventoryItem, InventoryDelta> resource_deltas;  // Per-tick resource changes
  std::vector<int> target_tag_ids;  // If non-empty, only affect objects with these tags
  bool members_only = false;        // Only affect objects with matching commons
  bool ignore_members = false;      // Ignore objects with matching commons

  AOEEffectConfig() = default;
  AOEEffectConfig(unsigned int range,
                  const std::unordered_map<InventoryItem, InventoryDelta>& resource_deltas,
                  const std::vector<int>& target_tag_ids = {},
                  bool members_only = false,
                  bool ignore_members = false)
      : range(range),
        resource_deltas(resource_deltas),
        target_tag_ids(target_tag_ids),
        members_only(members_only),
        ignore_members(ignore_members) {}
};

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;
  std::vector<int> tag_ids;
  ObservationType initial_vibe;
  std::optional<AOEEffectConfig> aoe;  // If set, object emits AOE effects

  GridObjectConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : type_id(type_id), type_name(type_name), tag_ids({}), initial_vibe(initial_vibe), aoe(std::nullopt) {}

  virtual ~GridObjectConfig() = default;
};

// Forward declaration for Alignable (to check commons membership)
class Alignable;

// Forward declaration for HasInventory
class HasInventory;

// Helper class for managing AOE effects on grid objects
class AOEHelper {
public:
  AOEHelper() = default;

  // Initialize with grid reference and owner
  void init(Grid* grid, GridObject* owner) {
    _grid = grid;
    _owner = owner;
  }

  // Set the AOE config (call from object constructor)
  void set_config(const AOEEffectConfig* config) {
    _config = config;
  }

  // Check if this helper has AOE configured
  bool has_aoe() const {
    return _config != nullptr && _grid != nullptr;
  }

  // Register AOE effects at the given location
  void register_effects(GridCoord r, GridCoord c);

  // Unregister AOE effects (call on demolish or removal)
  void unregister_effects();

  // Try to register an inventory object with this AOE
  // Returns true if the object was registered (passes all filters)
  // Called when a static HasInventory object is placed within range or alignment changes
  bool try_register_inventory_object(GridObject* obj);

  // Unregister an inventory object (when it's removed from the grid)
  void unregister_inventory_object(GridObject* obj);

  // Re-evaluate all registered objects (call when alignment changes might affect filtering)
  void refresh_registrations();

  // Get the set of registered static inventory objects (already filtered)
  const std::vector<GridObject*>& registered_inventory_objects() const {
    return _registered_inventory_objects;
  }

  // Check if an object matches the target_tag_ids filter
  // Returns true if target_tag_ids is empty or if the object has at least one matching tag
  bool matches_target_tags(const GridObject* obj) const;

  // Check if an object passes all AOE filters (target_tags + commons)
  // tag_id_map is needed for commons tag lookup on non-Alignable objects
  bool passes_all_filters(GridObject* obj, const std::unordered_map<int, std::string>& tag_id_map) const;

  // Get the config
  const AOEEffectConfig* config() const {
    return _config;
  }

  // Get the owner object
  GridObject* owner() const {
    return _owner;
  }

  // Check if this AOE is currently registered
  bool is_registered() const {
    return _registered;
  }

  // Get the center location
  GridCoord location_r() const {
    return _location_r;
  }
  GridCoord location_c() const {
    return _location_c;
  }

private:
  const AOEEffectConfig* _config = nullptr;
  Grid* _grid = nullptr;
  GridObject* _owner = nullptr;  // The object that owns this AOE effect
  bool _registered = false;
  GridCoord _location_r = 0;
  GridCoord _location_c = 0;
  std::vector<GridObject*> _registered_inventory_objects;  // Static objects affected by this AOE
};

class GridObject : public HasVibe {
private:
  std::optional<AOEEffectConfig> _aoe_config;

public:
  GridObjectId id{};
  GridLocation location{};
  TypeId type_id{};
  std::string type_name;
  std::vector<int> tag_ids;
  AOEHelper aoe;  // AOE effect helper

  virtual ~GridObject() = default;

  void init(TypeId object_type_id,
            const std::string& object_type_name,
            const GridLocation& object_location,
            const std::vector<int>& tags,
            ObservationType object_vibe = 0,
            const std::optional<AOEEffectConfig>& aoe_config = std::nullopt) {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->location = object_location;
    this->tag_ids = tags;
    this->vibe = object_vibe;
    if (aoe_config.has_value()) {
      _aoe_config = aoe_config.value();
      this->aoe.set_config(&_aoe_config.value());
    }
  }

  // Called when this object is removed. Override for cleanup.
  virtual void on_remove() {
    aoe.unregister_effects();
  }

  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
