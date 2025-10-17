#include <csignal>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "rpc/game_registry.hpp"
#include "rpc/socket_server.hpp"

namespace {

mettagrid::rpc::SocketServer* g_server = nullptr;

void HandleSignal(int /*signal_number*/) {
  if (g_server != nullptr) {
    g_server->Stop();
  }
}

struct ParsedArgs {
  std::string host = "127.0.0.1";
  uint16_t port = 5858;
};

ParsedArgs ParseArgs(int argc, char** argv) {
  ParsedArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string_view token(argv[i]);
    auto ExpectValue = [&](const char* flag, std::string_view remainder) {
      if (!remainder.empty()) {
        return std::string(remainder);
      }
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return std::string(argv[++i]);
    };

    if (token.rfind("--host", 0) == 0) {
      std::string value = ExpectValue("--host", token.size() > 7 ? token.substr(7) : std::string_view{});
      args.host = value;
    } else if (token.rfind("--port", 0) == 0) {
      std::string value = ExpectValue("--port", token.size() > 7 ? token.substr(7) : std::string_view{});
      int parsed = std::stoi(value);
      if (parsed < 0 || parsed > 65535) {
        throw std::runtime_error("port must be in [0, 65535]");
      }
      args.port = static_cast<uint16_t>(parsed);
    } else if (token == "--help" || token == "-h") {
      std::cout << "Usage: " << argv[0] << " [--host HOST] [--port PORT]\n";
      std::cout << "Defaults: host=127.0.0.1, port=5858\n";
      std::exit(0);
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + std::string(token));
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ParsedArgs args = ParseArgs(argc, argv);

    mettagrid::rpc::GameRegistry registry;
    mettagrid::rpc::ServerOptions options;
    options.host = args.host;
    options.port = args.port;

    mettagrid::rpc::SocketServer server(options);
    g_server = &server;
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    auto status = server.Serve(&registry);
    g_server = nullptr;

    if (!status.ok) {
      std::cerr << "Server exited with error: " << status.message << std::endl;
      return 1;
    }

    std::cout << "Server stopped." << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Fatal error: " << ex.what() << std::endl;
    return 1;
  }
}
