#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

// ============================================================================
// FUNDAMENTAL TYPE DEFINITIONS
// ============================================================================

// Grid object identifier - unique ID for each object in the grid
typedef uint32_t GridObjectId;

// Layer within the grid - used to stack objects
typedef uint16_t Layer;

// Type identifier for grid objects - corresponds to ObjectType enum
typedef uint16_t TypeId;

// Grid coordinate - position in the grid
typedef uint32_t GridCoord;

// Event identifier for the event system
typedef uint16_t EventId;

// Event argument - parameter data passed with events
typedef int32_t EventArg;

// ============================================================================
// NUMPY-COMPATIBLE TYPE DEFINITIONS
// ============================================================================

// Define core types used for numpy arrays
typedef uint8_t numpy_bool_t;
typedef uint8_t c_observations_type;
typedef numpy_bool_t c_terminals_type;
typedef numpy_bool_t c_truncations_type;
typedef float c_rewards_type;
typedef uint8_t c_actions_type;
typedef numpy_bool_t c_masks_type;
typedef numpy_bool_t c_success_type;

// ============================================================================
// NUMPY TYPE NAME MACROS
// ============================================================================

// Type names to use in Python - these must match the C++ types above
#define NUMPY_OBSERVATIONS_TYPE "uint8"  // match c_observations_type
#define NUMPY_TERMINALS_TYPE "uint8"     // match c_terminals_type
#define NUMPY_TRUNCATIONS_TYPE "uint8"   // match c_truncations_type
#define NUMPY_REWARDS_TYPE "float32"     // match c_rewards_type
#define NUMPY_ACTIONS_TYPE "uint8"       // match c_actions_type
#define NUMPY_MASKS_TYPE "uint8"         // match c_masks_type
#define NUMPY_SUCCESS_TYPE "uint8"       // match c_success_type

// ============================================================================
// TYPE MAPPING FUNCTION
// ============================================================================

// Function to provide NumPy type information to Python
inline const char* get_numpy_type_name(const char* type_id) {
  if (strcmp(type_id, "observations") == 0) return NUMPY_OBSERVATIONS_TYPE;
  if (strcmp(type_id, "terminals") == 0) return NUMPY_TERMINALS_TYPE;
  if (strcmp(type_id, "truncations") == 0) return NUMPY_TRUNCATIONS_TYPE;
  if (strcmp(type_id, "rewards") == 0) return NUMPY_REWARDS_TYPE;
  if (strcmp(type_id, "actions") == 0) return NUMPY_ACTIONS_TYPE;
  if (strcmp(type_id, "masks") == 0) return NUMPY_MASKS_TYPE;
  if (strcmp(type_id, "success") == 0) return NUMPY_SUCCESS_TYPE;
  return "unknown";
}

// ============================================================================
// COMPOUND TYPES
// ============================================================================

// Configuration for action handlers
typedef std::map<std::string, c_actions_type> ActionConfig;

// Event structure
struct Event {
  uint32_t timestamp;
  EventId event_id;
  GridObjectId object_id;
  EventArg arg;

  bool operator<(const Event& other) const {
    return timestamp > other.timestamp;
  }
};

class GridLocation {
public:
  GridCoord r;
  GridCoord c;
  Layer layer;

  inline GridLocation(GridCoord r, GridCoord c, Layer layer) : r(r), c(c), layer(layer) {}
  inline GridLocation(GridCoord r, GridCoord c) : r(r), c(c), layer(0) {}
  inline GridLocation() : r(0), c(0), layer(0) {}
};

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Forward declarations for all relevant classes
class Grid;
class StatsTracker;
class EventManager;
class EventHandler;
class Converter;
class GridObject;
class Agent;
class ActionHandler;
class ProductionHandler;
class CoolDownHandler;
class CppMettaGrid;

#endif  // TYPES_HPP