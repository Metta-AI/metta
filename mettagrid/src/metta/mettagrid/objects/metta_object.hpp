#ifndef OBJECTS_METTA_OBJECT_HPP_
#define OBJECTS_METTA_OBJECT_HPP_

#include <map>
#include <string>

#include "../grid_object.hpp"

typedef std::map<std::string, int> ObjectConfig;

class MettaObject : public GridObject {
public:
  virtual bool swappable() const {
    return false;
  }
};

#endif  // OBJECTS_METTA_OBJECT_HPP_
