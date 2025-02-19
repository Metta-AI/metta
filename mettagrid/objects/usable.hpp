#ifndef USABLE_HPP
#define USABLE_HPP

#include <map>
#include <string>
#include "metta_object.hpp"
#include "agent.hpp"

class Usable : public MettaObject {
public:
    unsigned int use_cost;
    unsigned int cooldown;
    unsigned char ready;

    void init_usable(ObjectConfig cfg) {
        this->ready = 1;
        this->use_cost = cfg["use_cost"];
        this->cooldown = cfg["cooldown"];
    }

    virtual bool usable(const Agent *actor) {
        return this->ready;
    }

    virtual bool is_usable_type() {
        return true;
    }

    virtual void use(Agent *actor, float *rewards) = 0;
};

#endif
