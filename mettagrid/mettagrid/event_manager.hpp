#ifndef EVENT_MANAGER_HPP
#define EVENT_MANAGER_HPP

#include <cassert>
#include <queue>
#include <vector>

#include "types.hpp"

// Base event handler definition
class EventHandler {
protected:
  EventManager* event_manager;

public:
  EventHandler(EventManager* em) {
    this->event_manager = em;
  }

  virtual ~EventHandler() {}

  virtual void handle_event(GridObjectId object_id, EventArg arg) = 0;
};

// Event Manager definition
class EventManager {
private:
  std::priority_queue<Event> _event_queue;
  uint32_t _current_timestep;

public:
  Grid* grid;
  StatsTracker* stats;
  std::vector<EventHandler*> event_handlers;

  EventManager() {
    this->grid = nullptr;
    this->stats = nullptr;
    this->_current_timestep = 0;
  }

  void init(Grid* grid, StatsTracker* stats) {
    this->grid = grid;
    this->stats = stats;
  }

  ~EventManager() {
    for (auto handler : this->event_handlers) {
      delete handler;
    }
  }

  void schedule_event(EventId event_id, uint32_t delay, GridObjectId object_id, EventArg arg) {
    Event event;
    // If the object id is 0, the object has probably not been added to the grid yet.
    assert(object_id != 0);
    event.timestamp = this->_current_timestep + delay;
    event.event_id = event_id;
    event.object_id = object_id;
    event.arg = arg;
    this->_event_queue.push(event);
  }

  void process_events(uint32_t current_timestep) {
    this->_current_timestep = current_timestep;
    Event event;
    while (!this->_event_queue.empty()) {
      event = this->_event_queue.top();
      if (event.timestamp > this->_current_timestep) {
        break;
      }
      this->_event_queue.pop();
      this->event_handlers[event.event_id]->handle_event(event.object_id, event.arg);
    }
  }
};

#endif  // EVENT_MANAGER_HPP