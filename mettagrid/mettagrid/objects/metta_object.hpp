#ifndef METTAGRID_METTAGRID_OBJECTS_METTA_OBJECT_HPP_
#define METTAGRID_METTAGRID_OBJECTS_METTA_OBJECT_HPP_

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

#endif  // METTAGRID_METTAGRID_OBJECTS_METTA_OBJECT_HPP_
