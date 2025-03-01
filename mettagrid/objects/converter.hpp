#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include <vector>
#include <string>
#include <cassert>
#include "../grid_object.hpp"
#include "../event.hpp"
#include "constants.hpp"
#include "metta_object.hpp"
#include "has_inventory.hpp"
#include "agent.hpp"

class Converter : public HasInventory {
private:
    // This should be called any time the converter could start converting. E.g.,
    // when things are added to its input, and when it finishes converting.
    void maybe_start_converting() {
        // We can't start converting if there's no event manager, since we won't
        // be able to schedule the finishing event.
        assert(this->event_manager != nullptr);
        // We also need to have an id to schedule the finishing event. If our id
        // is zero, we probably haven't been added to the grid yet.
        assert(this->id != 0);
        if (this->converting) {
            return;
        }
        // Check if the converter is already at max output.
        unsigned short total_output = 0;
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            if (this->recipe_output[i] > 0) {
                total_output += this->inventory[i];
            }
        }
        if (total_output >= this->max_output) {
            return;
        }
        // Check if the converter has enough input.
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            if (this->inventory[i] < this->recipe_input[i]) {
                return;
            }
        }
        // produce.
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            this->inventory[i] -= this->recipe_input[i];
        }
        // All the previous returns were "we don't start converting".
        // This one is us starting to convert.
        this->converting = true;
        this->event_manager->schedule_event(Events::FinishConverting, this->recipe_duration, this->id, 0);
    }

public:
    vector<unsigned char> recipe_input;
    vector<unsigned char> recipe_output;
    // The converter won't convert if its output already has this many things of
    // the type it produces. This may be clunky in some cases, but the main usage
    // is to make Mines (etc) have a maximum output.
    unsigned short max_output;
    unsigned char recipe_duration;
    bool converting;
    EventManager *event_manager;

    Converter(GridCoord r, GridCoord c, ObjectConfig cfg, TypeId type_id) {
        GridObject::init(type_id, GridLocation(r, c, GridLayer::Object_Layer));
        MettaObject::init_mo(cfg);
        HasInventory::init_has_inventory(cfg);
        this->recipe_input.resize(InventoryItem::InventoryCount);
        this->recipe_output.resize(InventoryItem::InventoryCount);
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            this->recipe_input[i] = cfg["input_" + InventoryItemNames[i]];
            this->recipe_output[i] = cfg["output_" + InventoryItemNames[i]];
        }
        this->max_output = cfg["max_output"];
        this->recipe_duration = cfg["cooldown"];
        this->converting = false;
    }

    Converter(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::GenericConverterT) {}

    void set_event_manager(EventManager *event_manager) {
        this->event_manager = event_manager;
        this->maybe_start_converting();
    }

    void finish_converting() {
        this->converting = false;
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            this->update_inventory(static_cast<InventoryItem>(i), this->recipe_output[i], nullptr);
        }
    }

    void update_inventory(InventoryItem item, short amount, float *reward) override {
        HasInventory::update_inventory(item, amount, reward);
        this->maybe_start_converting();
    }

    void obs(ObsType *obs, const std::vector<unsigned int> &offsets) const override {
        obs[offsets[0]] = 1;
        obs[offsets[1]] = this->hp;
        obs[offsets[2]] = this->converting;
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            obs[offsets[3] + i] = this->inventory[i];
        }
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        // We use the same feature names for all converters, since this compresses
        // the observation space. At the moment we don't expose the recipe, since
        // we expect converters to be hard coded.
        names.push_back("converter");
        names.push_back("hp");
        names.push_back("converting");
        for (unsigned int i = 0; i < InventoryItem::InventoryCount; i++) {
            names.push_back("inv:" + InventoryItemNames[i]);
        }
        return names;
    }
};

#endif
