#ifndef ALTAR_HPP
#define ALTAR_HPP

#include <vector>
#include <string>
#include "../grid_object.hpp"
#include "agent.hpp"
#include "constants.hpp"
#include "converter.hpp"

class Altar : public Converter {
public:
    Altar(GridCoord r, GridCoord c, ObjectConfig cfg) : Converter(r, c, cfg, ObjectType::AltarT) {}

    static std::vector<std::string> feature_names() {
        auto names = Converter::feature_names();
        names[0] = "altar";
        return names;
    }
};

#endif
