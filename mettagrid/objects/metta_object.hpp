#ifndef METTA_OBJECT_HPP
#define METTA_OBJECT_HPP

#include <map>
#include <string>
#include "../grid_object.hpp"

typedef std::map<std::string, int> ObjectConfig;

class MettaObject : public GridObject {
public:
    unsigned int hp;

    void init_mo(ObjectConfig cfg) {
        this->hp = cfg["hp"];
    }

    virtual bool is_usable_type() {
        return false;
    }
};

#endif
