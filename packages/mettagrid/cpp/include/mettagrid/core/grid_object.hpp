#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/aoe_config.hpp"
#include "core/types.hpp"
#include "handler/handler_config.hpp"
#include "objects/alignable.hpp"
#include "objects/constants.hpp"
#include "objects/has_inventory.hpp"
#include "objects/has_vibe.hpp"
#include "objects/inventory_config.hpp"
#include "objects/usable.hpp"

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

struct GridObjectConfig {
  TypeId type_id;
  std::string type_name;
  std::string name;  // Instance name (defaults to type_name if empty)
  std::vector<int> tag_ids;
  ObservationType initial_vibe;
  InventoryConfig inventory_config;
  std::unordered_map<InventoryItem, InventoryQuantity> initial_inventory;
  std::vector<std::shared_ptr<mettagrid::AOEConfig>> aoes;  // AOE effects emitted by this object
  std::vector<mettagrid::HandlerConfig> handlers;           // Handlers for this object

  GridObjectConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : type_id(type_id),
        type_name(type_name),
        name(""),
        tag_ids({}),
        initial_vibe(initial_vibe),
        inventory_config(),
        aoes(),
        handlers() {}

  virtual ~GridObjectConfig() = default;
};

// Forward declaration for Handler
namespace mettagrid {
class Handler;
}

class GridObject : public HasVibe, public Alignable, public HasInventory, public Usable {
public:
  GridObjectId id{};
  GridLocation location{};
  TypeId type_id{};
  std::string type_name;  // Class type (e.g., "assembler")
  std::string name;       // Instance name (e.g., "carbon_extractor"), defaults to type_name
  std::vector<int> tag_ids;

  // Constructor with optional inventory config (defaults to empty)
  explicit GridObject(const InventoryConfig& inv_config = InventoryConfig()) : HasInventory(inv_config) {}

  ~GridObject() override = default;

  void init(TypeId object_type_id,
            const std::string& object_type_name,
            const GridLocation& object_location,
            const std::vector<int>& tags,
            ObservationType object_vibe = 0,
            const std::string& object_name = "") {
    this->type_id = object_type_id;
    this->type_name = object_type_name;
    this->name = object_name.empty() ? object_type_name : object_name;
    this->location = object_location;
    this->tag_ids = tags;
    this->vibe = object_vibe;
  }

  // Set handlers from config
  void set_handlers(std::vector<std::shared_ptr<mettagrid::Handler>> handlers) {
    _handlers = std::move(handlers);
  }

  // Check if this object has any handlers
  bool has_handlers() const {
    return !_handlers.empty();
  }

  // Override onUse to try handlers
  bool onUse(Agent& actor, ActionArg arg) override;

  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
  }

protected:
  std::vector<std::shared_ptr<mettagrid::Handler>> _handlers;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
