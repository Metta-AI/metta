#ifndef METTA_OBJECT_HPP
#define METTA_OBJECT_HPP

#include <cstdint>
#include <map>
#include <string>

#include "grid_object.hpp"

typedef std::map<std::string, int32_t> ObjectConfig;

class MettaObject : public GridObject {
public:
  uint32_t hp;

  void set_hp(ObjectConfig cfg) {
    this->hp = cfg["hp"];
  }

  virtual void obs(ObsType* obs) const override {
    encode(obs, GridFeature::HP, this->hp);
  }

  virtual bool has_inventory() {  // TODO: make const
    return false;
  }

  virtual bool swappable() const {
    return false;
  }
};

#endif