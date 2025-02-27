#ifndef HAS_INVENTORY_HPP
#define HAS_INVENTORY_HPP

#include <map>
#include <string>
#include "metta_object.hpp"
#include "constants.hpp"

class HasInventory : public MettaObject {
public:
    vector<unsigned char> inventory;

    void init_has_inventory(ObjectConfig cfg) {
        this->inventory.resize(InventoryItem::InventoryCount);
    }

    virtual bool has_inventory() {
        return true;
    }

    // Whether the inventory is accessible to an agent.
    virtual bool inventory_is_accessible() {
        return true;
    }
};

#endif