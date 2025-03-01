#ifndef PRODUCTION_HANDLER_HPP
#define PRODUCTION_HANDLER_HPP

#include "converter.hpp"
#include "grid.hpp"
#include "stats_tracker.hpp"
#include "constants.hpp"
#include "event.hpp"
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

        if (converter->maybe_start_converting()) {
            this->event_manager->schedule_event(Events::FinishConverting, converter->recipe_duration, converter->id, 0);
        }
    }
};


#endif // PRODUCTION_HANDLER_HPP