#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_

#include "core/event.hpp"
#include "core/grid.hpp"
#include "objects/assembler.hpp"
#include "objects/constants.hpp"

// Handles the FinishConverting event
class ProductionHandler : public EventHandler {
public:
  explicit ProductionHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    // Converter handling removed
    // This handler is now primarily for Assembler production events
    (void)obj_id;  // Suppress unused parameter warning
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

    // Converter cooldown handling removed
    // NB: Assemblers handle cooldown implicitly.
    (void)obj;  // Suppress unused parameter warning
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
