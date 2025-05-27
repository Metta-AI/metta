#ifndef EVENT_H
#define EVENT_H

#include <cassert>
#include <map>
#include <queue>
using namespace std;

#include "grid.hpp"
#include "grid_object.hpp"
#include "stats_tracker.hpp"
typedef int EventArg;
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
  EventHandler(EventManager* em) {
    this->event_manager = em;
  }

  virtual ~EventHandler() {}

  virtual void handle_event(GridObjectId object_id, EventArg arg) = 0;
};

class EventManager {
private:
  priority_queue<Event> _event_queue;
  unsigned int _current_timestep;

public:
  Grid* grid;
  StatsTracker* stats;
  map<EventType, unique_ptr<EventHandler>> event_handlers;

  EventManager() {
    this->grid = nullptr;
    this->stats = nullptr;
  }

  void init(Grid* grid, StatsTracker* stats) {
    this->grid = grid;
    this->stats = stats;
    this->_current_timestep = 0;
  }

  void schedule_event(EventType event_type, unsigned int delay, GridObjectId object_id, EventArg arg) {
    Event event;
    // If the object id is 0, the object has probably not been added to the grid yet. Given
    // our current usage of events, this is an error, since we won't be able to find the object
    // later when the event resolves.
    assert(object_id != 0);
    event.timestamp = this->_current_timestep + delay;
    event.event_type = event_type;
    event.object_id = object_id;
    event.arg = arg;
    this->_event_queue.push(event);
  }

  void process_events(unsigned int current_timestep) {
    this->_current_timestep = current_timestep;
    Event event;
    while (!this->_event_queue.empty()) {
      event = this->_event_queue.top();
      if (event.timestamp > this->_current_timestep) {
        break;
      }
      this->_event_queue.pop();
      this->event_handlers[event.event_type]->handle_event(event.object_id, event.arg);
    }
  }
};

#endif  // EVENT_H
