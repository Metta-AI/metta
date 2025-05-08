#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

// Basic type definitions
typedef uint32_t GridObjectId;
typedef uint8_t ObsType;
typedef uint16_t Layer;
typedef uint16_t TypeId;
typedef uint32_t GridCoord;
typedef uint16_t EventId;
typedef int32_t EventArg;
typedef uint32_t ActionType;
typedef std::map<std::string, int32_t> ActionConfig;

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