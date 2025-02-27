#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Factory : public Converter {
public:
    Factory(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::FactoryT) {}

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "factory";
        return names;
    }
};

#endif
