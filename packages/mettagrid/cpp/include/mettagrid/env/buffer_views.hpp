#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_BUFFER_VIEWS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_BUFFER_VIEWS_HPP_

#include <cstddef>
#include <algorithm>
#include <span>
#include <type_traits>

#include "core/grid_object.hpp"
#include "core/types.hpp"

// Lightweight contiguous buffer views used by the core engine. These avoid any
// dependency on pybind11 containers so the gameplay loop can run in pure C++.

namespace mettagrid::env {

template <typename T>
struct ArrayView {
  T* data{nullptr};
  size_t size{0};

  constexpr bool empty() const noexcept {
    return data == nullptr || size == 0;
  }

  constexpr T& operator[](size_t index) const noexcept {
    return data[index];
  }

  constexpr T* begin() const noexcept {
    return data;
  }

  constexpr T* end() const noexcept {
    return data + size;
  }

  void fill(const T& value) const {
    if (empty()) {
      return;
    }
    std::fill_n(data, size, value);
  }
};

template <typename T>
struct MatrixView {
  T* data{nullptr};
  size_t rows{0};
  size_t cols{0};

  constexpr bool empty() const noexcept {
    return data == nullptr || rows == 0 || cols == 0;
  }

  constexpr T* row(size_t r) const noexcept {
    return data + r * cols;
  }

  constexpr const T* crow(size_t r) const noexcept {
    return data + r * cols;
  }
};

struct ObservationBufferView {
  ObservationType* data{nullptr};
  size_t num_agents{0};
  size_t tokens_per_agent{0};
  size_t components_per_token{0};

  constexpr bool empty() const noexcept {
    return data == nullptr || num_agents == 0 || tokens_per_agent == 0 || components_per_token == 0;
  }

  constexpr size_t total_elements() const noexcept {
    return num_agents * tokens_per_agent * components_per_token;
  }

  constexpr ObservationToken* tokens(size_t agent_idx, size_t token_offset = 0) const noexcept {
    auto base = data + agent_idx * tokens_per_agent * components_per_token;
    return reinterpret_cast<ObservationToken*>(base + token_offset * components_per_token);
  }
};

struct BufferSet {
  ObservationBufferView observations;
  ArrayView<TerminalType> terminals;
  ArrayView<TruncationType> truncations;
  ArrayView<RewardType> rewards;
  ArrayView<RewardType> episode_rewards;
};

}  // namespace mettagrid::env

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_BUFFER_VIEWS_HPP_
