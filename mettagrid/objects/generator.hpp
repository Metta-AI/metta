#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Generator : public Converter {
public:
    Generator(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::GeneratorT) {
        this->recipe_input[InventoryItem::ore] = 1;
        this->recipe_output[InventoryItem::battery] = 1;
        this->recipe_duration = cfg["cooldown"];
    }

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "generator";
        return names;
    }
};

#endif
