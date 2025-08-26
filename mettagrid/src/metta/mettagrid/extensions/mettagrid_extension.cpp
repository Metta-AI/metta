// extensions/mettagrid_extension.cpp
#include "extensions/mettagrid_extension.hpp"

#include "mettagrid_c.hpp"

py::array_t<uint8_t>& MettaGridExtension::getObservations(MettaGrid* env) {
  return env->_observations;
}

const py::array_t<uint8_t>& MettaGridExtension::getObservations(const MettaGrid* env) const {
  return env->_observations;
}
