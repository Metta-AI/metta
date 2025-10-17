#include "rpc/socket_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <memory>
#include <stdexcept>

#include "rpc/status.hpp"

namespace mettagrid::rpc {

SocketServer::SocketServer(ServerOptions options) : options_(std::move(options)) {}

SocketServer::~SocketServer() {
  Stop();
}

Status SocketServer::Serve(GameRegistry* registry) {
  if (registry == nullptr) {
    return Status::Error("registry is null");
  }

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    return Status::Error(std::string("socket() failed: ") + std::strerror(errno));
  }

  if (options_.reuse_address) {
    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(options_.port);
  if (::inet_pton(AF_INET, options_.host.c_str(), &addr.sin_addr) != 1) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return Status::Error("invalid bind address");
  }

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    Status status = Status::Error(std::string("bind() failed: ") + std::strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return status;
  }

  if (::listen(listen_fd_, options_.backlog) < 0) {
    Status status = Status::Error(std::string("listen() failed: ") + std::strerror(errno));
    ::close(listen_fd_);
    listen_fd_ = -1;
    return status;
  }

  shutting_down_.store(false);
  Status accept_status = AcceptLoop(registry);
  Stop();
  return accept_status;
}

void SocketServer::Stop() {
  shutting_down_.store(true);
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

Status SocketServer::AcceptLoop(GameRegistry* registry) {
  while (!shutting_down_.load()) {
    int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (shutting_down_.load()) {
        break;
      }
      return Status::Error(std::string("accept() failed: ") + std::strerror(errno));
    }

    workers_.emplace_back([this, client_fd, registry]() {
      HandleClient(client_fd, registry);
    });
  }
  return Status::Ok();
}

void SocketServer::HandleClient(int client_fd, GameRegistry* registry) {
  // RAII guard to ensure file descriptor is closed on exit
  struct FdGuard {
    int fd;
    explicit FdGuard(int f) : fd(f) {}
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
  };
  FdGuard fd_guard(client_fd);

  std::string buffer;
  while (!shutting_down_.load() && ReadMessage(client_fd, &buffer)) {
    v1::MettaGridRequest request;
    if (!request.ParseFromString(buffer)) {
      v1::MettaGridResponse response;
      auto* error = response.mutable_error();
      error->set_code(1);
      error->set_message("failed to parse request");
      std::string out;
      response.SerializeToString(&out);
      WriteMessage(client_fd, out);
      continue;
    }

    v1::MettaGridResponse response;
    response.set_request_id(request.request_id());

    auto dispatch = [&](const Status& status) {
      if (!status.ok) {
        auto* error = response.mutable_error();
        error->set_code(1);
        error->set_message(status.message);
      }
    };

    if (request.has_create_game()) {
      Status status = registry->CreateGame(request.create_game());
      auto* ack = response.mutable_create_result();
      ack->set_ok(status.ok);
      if (!status.ok) {
        ack->set_message(status.message);
      }
    } else if (request.has_step_game()) {
      Status status = registry->StepGame(request.step_game(), response.mutable_step_result());
      if (!status.ok) {
        response.clear_step_result();
        dispatch(status);
      }
    } else if (request.has_get_state()) {
      Status status = registry->GetState(request.get_state(), response.mutable_state_result());
      if (!status.ok) {
        response.clear_state_result();
        dispatch(status);
      }
    } else if (request.has_delete_game()) {
      Status status = registry->DeleteGame(request.delete_game());
      auto* ack = response.mutable_delete_result();
      ack->set_ok(status.ok);
      if (!status.ok) {
        ack->set_message(status.message);
      }
    } else {
      dispatch(Status::Error("request payload is empty"));
    }

    std::string out;
    response.SerializeToString(&out);
    if (!WriteMessage(client_fd, out).ok) {
      break;
    }
  }
}

bool SocketServer::ReadMessage(int fd, std::string* buffer) {
  uint32_t length_be = 0;
  ssize_t read_bytes = ::recv(fd, &length_be, sizeof(length_be), MSG_WAITALL);
  if (read_bytes == 0) {
    return false;  // connection closed
  }
  if (read_bytes < 0) {
    return false;
  }
  uint32_t length = ntohl(length_be);
  if (length == 0) {
    buffer->clear();
    return true;
  }
  buffer->resize(length);
  ssize_t total = 0;
  char* data = buffer->data();
  while (total < static_cast<ssize_t>(length)) {
    ssize_t chunk = ::recv(fd, data + total, length - total, 0);
    if (chunk <= 0) {
      return false;
    }
    total += chunk;
  }
  return true;
}

Status SocketServer::WriteMessage(int fd, const std::string& buffer) {
  uint32_t length = static_cast<uint32_t>(buffer.size());
  uint32_t length_be = htonl(length);
  if (::send(fd, &length_be, sizeof(length_be), 0) != sizeof(length_be)) {
    return Status::Error(std::string("send() failed: ") + std::strerror(errno));
  }
  size_t total = 0;
  while (total < buffer.size()) {
    ssize_t written = ::send(fd, buffer.data() + total, buffer.size() - total, 0);
    if (written <= 0) {
      return Status::Error(std::string("send() failed: ") + std::strerror(errno));
    }
    total += static_cast<size_t>(written);
  }
  return Status::Ok();
}

}  // namespace mettagrid::rpc
