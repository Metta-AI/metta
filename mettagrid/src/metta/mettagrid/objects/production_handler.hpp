#ifndef OBJECTS_PRODUCTION_HANDLER_HPP_
#define OBJECTS_PRODUCTION_HANDLER_HPP_

#include "../event.hpp"
#include "../grid.hpp"
#include "constants.hpp"
#include "converter.hpp"
#include "nano_assembler.hpp"

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
    GridObject* obj = this->event_manager->grid->object(obj_id);
    if (!obj) {
      return;
    }

    // Handle Converter cooldown
    Converter* converter = dynamic_cast<Converter*>(obj);
    if (converter) {
      converter->finish_cooldown();
      return;
    }

    // Handle NanoAssembler cooldown
    NanoAssembler* nano_assembler = dynamic_cast<NanoAssembler*>(obj);
    if (nano_assembler) {
      nano_assembler->finish_cooldown();
      return;
    }
  }
};

#endif  // OBJECTS_PRODUCTION_HANDLER_HPP_
