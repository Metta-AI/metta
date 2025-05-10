#ifndef EVENT_HANDLERS_HPP
#define EVENT_HANDLERS_HPP

#include "event_manager.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_converting();
    this->event_manager->stats->incr(ObjectTypeNames[converter->_type_id], "produced");
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

#endif  // EVENT_HANDLERS_HPP