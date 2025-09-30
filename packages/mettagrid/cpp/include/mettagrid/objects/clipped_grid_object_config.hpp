// clipped_grid_object_config.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_CONFIG_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_CONFIG_HPP_

#include <string>
#include <vector>

#include "objects/assembler_config.hpp"

struct ClippedGridObjectConfig : public AssemblerConfig {
  ClippedGridObjectConfig(TypeId type_id, const std::string& type_name, const std::vector<int>& tag_ids = {})
      : AssemblerConfig(type_id, type_name, tag_ids) {}
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_CLIPPED_GRID_OBJECT_CONFIG_HPP_
