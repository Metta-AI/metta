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
    cout << "production handler: handling event" << endl;
    auto grid = this->event_manager->grid;
    cout << "production handler: grid: " << grid << endl;
    // is grid null?
    if (!grid) {
      cout << "production handler: grid is null" << endl;
      return;
    }
    // print grid dimensions
    cout << "production handler: grid object count: " << grid->objects.size() << endl;
    cout << "production handler: grid dimensions: " << grid->width << "x" << grid->height << endl;
    // print grid object count
    
    cout << "production handler: object id: " << obj_id << endl;
    cout << "getting object" << endl;
    auto obj = grid->objects[obj_id];
    cout << "production handler: object: " << obj << endl;
    Converter* converter = static_cast<Converter*>(obj);
    cout << "production handler: converter: " << converter << endl;
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
