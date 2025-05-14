#ifndef EVENT_MANAGER_HPP
#define EVENT_MANAGER_HPP

#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

#include "types.hpp"

// Event Manager definition
class EventManager {
private:
  std::priority_queue<Event> _event_queue;
  uint32_t _current_timestep;
  std::vector<std::function<void(GridObjectId, EventArg)>> _event_handlers;

public:
  Grid* grid;
  StatsTracker* stats;

  EventManager() : _current_timestep(0), grid(nullptr), stats(nullptr) {}

  void init(Grid* grid, StatsTracker* stats) {
    this->grid = grid;
    this->stats = stats;

    // Pre-size the handler vector to accommodate all event types
    // You might want to use a constant like Events::Count or Events::MAX_EVENT
    _event_handlers.resize(10);  // Adjust this to a suitable size
  }

  // Register an event handler for a specific event type
  template <typename F>
  void register_handler(EventId event_id, F&& handler) {
    // Resize if needed
    if (event_id >= _event_handlers.size()) {
      _event_handlers.resize(event_id + 1);
    }

    _event_handlers[event_id] = std::forward<F>(handler);
  }

  void schedule_event(EventId event_id, uint32_t delay, GridObjectId object_id, EventArg arg = 0) {
    // If the object id is 0, the object has probably not been added to the grid yet.
    assert(object_id != 0);

    // Make sure the event_id is within valid range
    assert(event_id < _event_handlers.size() && "Event ID out of range");
    assert(_event_handlers[event_id] && "No handler registered for this event");

    Event event;
    event.timestamp = _current_timestep + delay;
    event.event_id = event_id;
    event.object_id = object_id;
    event.arg = arg;

    _event_queue.push(event);
  }

  void process_events(uint32_t current_timestep) {
    _current_timestep = current_timestep;

    while (!_event_queue.empty()) {
      Event event = _event_queue.top();
      if (event.timestamp > _current_timestep) {
        break;
      }
      _event_queue.pop();

      // Check if the event_id is valid
      if (event.event_id < _event_handlers.size() && _event_handlers[event.event_id]) {
        _event_handlers[event.event_id](event.object_id, event.arg);
      }
    }
  }
};

// Convenience method to create a lambda that calls a method on an object
template <typename T, typename Method>
auto make_event_handler(EventManager* em, Method method) {
  return [em, method](GridObjectId object_id, EventArg arg) {
    // Get the object from the grid
    T* obj = static_cast<T*>(em->grid->object(object_id));
    if (obj) {
      (obj->*method)(arg);
    }
  };
}

#endif  // EVENT_MANAGER_HPP