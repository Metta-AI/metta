#ifndef OBJECTS_PRODUCTION_HANDLER_HPP_
#define OBJECTS_PRODUCTION_HANDLER_HPP_

#include "../event.hpp"
#include "../grid.hpp"
#include "constants.hpp"
#include "converter.hpp"

// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  explicit ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_converting();
    converter->stats.incr(converter->type_name + ".produced");
  }
};

// Handles the CoolDown event
class CoolDownHandler : public EventHandler {
public:
  explicit CoolDownHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_cooldown();
  }
};

// Handles the FinishPhase event
class PhaseHandler : public EventHandler {
public:
  explicit PhaseHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_phase();
  }
};

#endif  // OBJECTS_PRODUCTION_HANDLER_HPP_
