#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_PROTO_CONVERTERS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_PROTO_CONVERTERS_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/mettagrid_config_types.hpp"
#include "proto/mettagrid/rpc/v1/mettagrid_service.pb.h"

namespace mettagrid::rpc {

GameConfig ConvertGameConfig(const v1::GameConfig& proto_cfg);
std::vector<std::vector<std::string>> ConvertMapDefinition(const v1::MapDefinition& proto_map);

}  // namespace mettagrid::rpc

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_PROTO_CONVERTERS_HPP_
