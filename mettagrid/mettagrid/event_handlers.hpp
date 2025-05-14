#ifndef EVENT_HANDLERS_HPP
#define EVENT_HANDLERS_HPP

#include <functional>

#include "event_manager.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

namespace EventHandlers {

inline std::function<void(GridObjectId, EventArg)> create_production_handler(EventManager* event_manager) {
  return [event_manager](GridObjectId obj_id, EventArg arg) {
    Converter* converter = static_cast<Converter*>(event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_converting();
    event_manager->stats->incr(ObjectTypeNames[converter->_type_id], "produced");
  };
}

// Creates a handler for the CoolDown event
inline std::function<void(GridObjectId, EventArg)> create_cooldown_handler(EventManager* event_manager) {
  return [event_manager](GridObjectId obj_id, EventArg arg) {
    Converter* converter = static_cast<Converter*>(event_manager->grid->object(obj_id));
    if (!converter) {
      return;
    }

    converter->finish_cooldown();
  };
}

// Function to register all event handlers
inline void register_all(EventManager* event_manager) {
  event_manager->register_handler(Events::FinishConverting, create_production_handler(event_manager));
  event_manager->register_handler(Events::CoolDown, create_cooldown_handler(event_manager));
}

}  // namespace EventHandlers

#endif  // EVENT_HANDLERS_HPP