#ifndef METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
#define METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_

#include "../event.hpp"
#include "../grid.hpp"
#include "../stats_tracker.hpp"
#include "constants.hpp"
#include "converter.hpp"

// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  explicit ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

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
  explicit CoolDownHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    Converter* converter = static_cast<Converter*>(this->event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_cooldown();
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
