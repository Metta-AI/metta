#pragma once

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "constants.hpp"
#include "metta_object.hpp"
#include "usable.hpp"
#include "agent.hpp"

class Converter : public Usable {
public:
    short prey_r1_output_energy;
    short predator_r1_output_energy; 
    short predator_r2_output_energy;

    Converter(GridCoord r, GridCoord c, ObjectConfig cfg) {
        GridObject::init(ObjectType::ConverterT, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        Usable::init_usable(cfg);
        this->prey_r1_output_energy = cfg["energy_output.r1.prey"];
        this->predator_r1_output_energy = cfg["energy_output.r1.predator"];
        this->predator_r2_output_energy = cfg["energy_output.r2.predator"];
    }

    bool usable(const Agent* actor) {
        return Usable::usable(actor) && (
            actor->inventory[InventoryItem::r1] > 0 ||
            (actor->inventory[InventoryItem::r2] > 0 &&
             actor->group_name == "predator")
        );
    }

    void use(Agent* actor, unsigned int actor_id, float* rewards) {
        unsigned int energy_gain = 0;
        InventoryItem consumed_resource = InventoryItem::r1;
        InventoryItem produced_resource = InventoryItem::r2;
        unsigned int potential_energy_gain = this->prey_r1_output_energy;
        
        if (actor->group_name == "predator") {
            if (actor->inventory[InventoryItem::r2] > 0) {
                // eat meat if you can
                consumed_resource = InventoryItem::r2;
                produced_resource = InventoryItem::r3;
                potential_energy_gain = this->predator_r2_output_energy;
            } else {
                potential_energy_gain = this->predator_r1_output_energy;
                produced_resource = InventoryItem::r3;
            }
        }

        actor->update_inventory(consumed_resource, -1, nullptr);
        actor->stats.incr(InventoryItemNames[consumed_resource], "used");
        actor->stats.incr(InventoryItemNames[consumed_resource], actor->group_name, "used");

        actor->update_inventory(produced_resource, 1, nullptr);
        actor->stats.incr(InventoryItemNames[produced_resource], "gained");
        actor->stats.incr(InventoryItemNames[produced_resource], actor->group_name, "gained");

        energy_gain = actor->update_energy(potential_energy_gain, rewards);
        actor->stats.add("energy.gained", energy_gain);
        actor->stats.add("energy.gained", actor->group_name, energy_gain);
    }

    static std::vector<std::string> feature_names() {
        return {"converter", "converter:hp", "converter:ready"};
    }
};
