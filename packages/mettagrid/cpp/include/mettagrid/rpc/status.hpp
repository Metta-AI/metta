#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_STATUS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_STATUS_HPP_

#include <string>

namespace mettagrid::rpc {

struct Status {
  bool ok;
  std::string message;

  static Status Ok() {
    return Status{true, ""};
  }

  static Status Error(std::string msg) {
    return Status{false, std::move(msg)};
  }
};

}  // namespace mettagrid::rpc

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_STATUS_HPP_
