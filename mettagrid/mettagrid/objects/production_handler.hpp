#ifndef METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
#define METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_

#include "../event.hpp"
#include "../grid.hpp"
#include "constants.hpp"
#include "converter.hpp"
#include "freeze_tower.hpp"

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
    converter->stats.incr(ObjectTypeNames[converter->_type_id] + ".produced");
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

// Handles the FreezeTowerAttack event
class FreezeTowerAttackHandler : public EventHandler {
public:
  explicit FreezeTowerAttackHandler(EventManager* event_manager) : EventHandler(event_manager) {}

  void handle_event(GridObjectId obj_id, EventArg arg) override {
    FreezeTower* freeze_tower = static_cast<FreezeTower*>(this->event_manager->grid->object(obj_id));
    if (!freeze_tower) {
      return;
    }

    freeze_tower->finish_cooldown();
  }
};

#endif  // METTAGRID_METTAGRID_OBJECTS_PRODUCTION_HANDLER_HPP_
