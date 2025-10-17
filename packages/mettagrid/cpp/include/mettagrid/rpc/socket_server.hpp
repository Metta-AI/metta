#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_SOCKET_SERVER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_SOCKET_SERVER_HPP_

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "rpc/game_registry.hpp"

namespace mettagrid::rpc {

struct ServerOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 5858;
  int backlog = 128;
  bool reuse_address = true;
};

class SocketServer {
public:
  explicit SocketServer(ServerOptions options);
  ~SocketServer();

  Status Serve(GameRegistry* registry);
  void Stop();

private:
  Status AcceptLoop(GameRegistry* registry);
  void HandleClient(int client_fd, GameRegistry* registry);

  bool ReadMessage(int fd, std::string* buffer);
  Status WriteMessage(int fd, const std::string& buffer);

  ServerOptions options_;
  int listen_fd_{-1};
  std::atomic<bool> shutting_down_{false};
  std::vector<std::thread> workers_;
};

}  // namespace mettagrid::rpc

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_RPC_SOCKET_SERVER_HPP_
