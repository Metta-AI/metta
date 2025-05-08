#ifndef PRODUCTION_HANDLER_HPP
#define PRODUCTION_HANDLER_HPP

#include "../event.hpp"
#include "../grid.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "converter.hpp"
#include <iostream>
// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    cout << "production handler: finishing conversion" << endl;
    converter->finish_converting();
    cout << "production handler: incrementing produced" << endl;
    this->event_manager->stats->incr(ObjectTypeNames[converter->_type_id], "produced");
    cout << "production handler: done" << endl;
  }
};

// Handles the CoolDown event
class CoolDownHandler : public EventHandler {
public:
  CoolDownHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_cooldown();
  }
};

#endif  // PRODUCTION_HANDLER_HPP
