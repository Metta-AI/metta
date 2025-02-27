#ifndef TEMPLE_HPP
#define TEMPLE_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Temple : public Converter {
public:
    Temple(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::TempleT) {}

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "temple";
        return names;
    }
};

#endif
