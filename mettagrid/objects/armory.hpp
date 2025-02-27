#ifndef ARMORY_HPP
#define ARMORY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Armory : public Converter {
public:
    Armory(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::ArmoryT) {}

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "armory";
        return names;
    }
};

#endif
