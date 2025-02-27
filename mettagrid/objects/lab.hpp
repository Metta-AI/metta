#ifndef LAB_HPP
#define LAB_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Lab : public Converter {
public:
    Lab(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::LabT) {
        this->recipe_input[InventoryItem::battery] = 3;
        this->recipe_input[InventoryItem::ore] = 3;
        this->recipe_output[InventoryItem::blueprint] = 1;
        this->recipe_duration = cfg["cooldown"];
    }

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "lab";
        return names;
    }
};

#endif
