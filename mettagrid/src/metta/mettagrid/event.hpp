#ifndef EVENT_HPP_
#define EVENT_HPP_

#include <cassert>
#include <map>
#include <memory>
#include <queue>

#include "grid.hpp"
#include "grid_object.hpp"

using std::map;
using std::priority_queue;
using std::unique_ptr;

using EventArg = int;

struct Event {
  unsigned int timestamp;
  EventType event_type;
  GridObjectId object_id;
  EventArg arg;

  bool operator<(const Event& other) const {
    return timestamp > other.timestamp;
  }
};

class EventManager;

class EventHandler {
protected:
  EventManager* event_manager;

public:
  explicit EventHandler(EventManager* em) : event_manager(em) {}
  virtual ~EventHandler() {}
  virtual void handle_event(GridObjectId object_id, EventArg arg) = 0;
};

class EventManager {
private:
  priority_queue<Event> _event_queue;
  unsigned int _current_timestep;

public:
  Grid* grid;
  map<EventType, unique_ptr<EventHandler>> event_handlers;

  EventManager() : grid(nullptr), _current_timestep(0) {}

  void init(Grid* grid) {
    this->grid = grid;
  }

  void schedule_event(EventType event_type, unsigned int delay, GridObjectId object_id, EventArg arg) {
    // If the object id is 0, the object has probably not been added to the grid yet. Given
    // our current usage of events, this is an error, since we won't be able to find the object
    // later when the event resolves.
    assert(object_id != 0);

    Event event;
    event.timestamp = this->_current_timestep + delay;
    event.event_type = event_type;
    event.object_id = object_id;
    event.arg = arg;

    this->_event_queue.push(event);
  }

  void process_events(unsigned int current_timestep) {
    this->_current_timestep = current_timestep;

    while (!this->_event_queue.empty()) {
      Event event = this->_event_queue.top();

      if (event.timestamp > this->_current_timestep) {
        break;
      }

      this->_event_queue.pop();
      this->event_handlers[event.event_type]->handle_event(event.object_id, event.arg);
    }
  }
};

#endif  // EVENT_HPP_
