#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

// Grid object identifier - unique ID for each object in the grid
typedef uint32_t GridObjectId;

// Observation data type - used for agent observations of the environment
// Limited to 8 bits to reduce memory footprint
typedef uint8_t ObsType;

// Layer within the grid - used to stack objects (e.g., agents on top of terrain)
typedef uint16_t Layer;

// Type identifier for grid objects - corresponds to ObjectType enum
typedef uint16_t TypeId;

// Grid coordinate - position in the grid (row or column)
typedef uint32_t GridCoord;

// Event identifier - used for the event system to identify different event types
typedef uint16_t EventId;

// Event argument - parameter data passed with events
typedef int32_t EventArg;

// Action identifier - used for the elements in the action array
// The "action id" is (actions[agent_idx][0])
// The "action arg" is (actions[agent_idx][1])
// This is a raw type that also corresponds to the ActionType enum values
typedef uint8_t ActionsType;

// Configuration for action handlers - maps setting names to integer values
typedef std::map<std::string, ActionsType> ActionConfig;

// match to numpy bool
typedef uint8_t numpy_bool_t;

// Forward declarations for all relevant classes
class Grid;
class StatsTracker;
class EventManager;
class EventHandler;
class Converter;
class Grid;
class GridObject;
class Agent;
class ActionHandler;
class ProductionHandler;
class CoolDownHandler;

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

#endif  // TYPES_HPP