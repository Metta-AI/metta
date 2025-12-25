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

// Forward declaration for AOEHelper
class Grid;

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

  AOEEffectConfig() = default;
  AOEEffectConfig(unsigned int range, const std::unordered_map<InventoryItem, InventoryDelta>& resource_deltas)
      : range(range), resource_deltas(resource_deltas) {}
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

// Helper class for managing AOE effects on grid objects
class AOEHelper {
public:
  AOEHelper() = default;

  // Initialize with grid reference
  void init(Grid* grid) {
    _grid = grid;
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

  // Get the config
  const AOEEffectConfig* config() const {
    return _config;
  }

private:
  const AOEEffectConfig* _config = nullptr;
  Grid* _grid = nullptr;
  bool _registered = false;
  GridCoord _location_r = 0;
  GridCoord _location_c = 0;
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
