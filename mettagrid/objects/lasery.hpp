#ifndef LASERY_HPP
#define LASERY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Lasery : public Converter {
public:
    Lasery(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::LaseryT) {
        this->recipe_input[InventoryItem::ore] = 1;
        this->recipe_input[InventoryItem::battery] = 2;
        this->recipe_output[InventoryItem::laser] = 1;
        this->recipe_duration = cfg["cooldown"];
    }

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "lasery";
        return names;
    }
};

#endif
