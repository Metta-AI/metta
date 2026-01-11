#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

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

  // Three types of handlers on GridObject:
  // - on_use: Triggered when agent uses/activates this object (context: actor=agent, target=this)
  // - on_update: Triggered after mutations are applied to this object (context: actor=null, target=this)
  // - aoe: Triggered per-tick for objects within radius (context: actor=this, target=affected)
  std::vector<mettagrid::HandlerConfig> on_use_handlers;
  std::vector<mettagrid::HandlerConfig> on_update_handlers;
  std::vector<mettagrid::HandlerConfig> aoe_handlers;

  GridObjectConfig(TypeId type_id, const std::string& type_name, ObservationType initial_vibe = 0)
      : type_id(type_id),
        type_name(type_name),
        name(""),
        tag_ids({}),
        initial_vibe(initial_vibe),
        inventory_config(),
        on_use_handlers(),
        on_update_handlers(),
        aoe_handlers() {}

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

  // Set handlers for each type
  void set_on_use_handlers(std::vector<std::shared_ptr<mettagrid::Handler>> handlers) {
    _on_use_handlers = std::move(handlers);
  }

  void set_on_update_handlers(std::vector<std::shared_ptr<mettagrid::Handler>> handlers) {
    _on_update_handlers = std::move(handlers);
  }

  void set_aoe_handlers(std::vector<std::shared_ptr<mettagrid::Handler>> handlers) {
    _aoe_handlers = std::move(handlers);
  }

  // Check if this object has any handlers of each type
  bool has_on_use_handlers() const {
    return !_on_use_handlers.empty();
  }

  bool has_on_update_handlers() const {
    return !_on_update_handlers.empty();
  }

  bool has_aoe_handlers() const {
    return !_aoe_handlers.empty();
  }

  // Get handlers for AOE processing
  const std::vector<std::shared_ptr<mettagrid::Handler>>& aoe_handlers() const {
    return _aoe_handlers;
  }

  // Override onUse to try on_use handlers
  bool onUse(Agent& actor, ActionArg arg) override;

  // Fire on_update handlers (called after mutations are applied)
  void fire_on_update_handlers();

  virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
  }

protected:
  std::vector<std::shared_ptr<mettagrid::Handler>> _on_use_handlers;
  std::vector<std::shared_ptr<mettagrid::Handler>> _on_update_handlers;
  std::vector<std::shared_ptr<mettagrid::Handler>> _aoe_handlers;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_GRID_OBJECT_HPP_
