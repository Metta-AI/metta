#ifndef MINE_HPP
#define MINE_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Mine : public Converter {
public:
    Mine(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::MineT) {
        this->recipe_output[InventoryItem::ore] = 1;
        this->recipe_duration = cfg["cooldown"];
    }

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "mine";
        return names;
    }
};

#endif
