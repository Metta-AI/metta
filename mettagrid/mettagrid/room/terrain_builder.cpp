#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <unordered_set>

namespace py = pybind11;

namespace {

// Direct numpy array access for better performance
std::vector<std::pair<int, int>> get_valid_positions(py::array_t<char> level) {
    auto r = level.unchecked<2>();
    int rows = r.shape(0);
    int cols = r.shape(1);
    std::vector<std::pair<int, int>> positions;
    positions.reserve((rows - 2) * (cols - 2)); // Pre-allocate for better performance

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (r(i, j) == 'e') { // 'e' for empty
                if (r(i-1, j) == 'e' || r(i+1, j) == 'e' ||
                    r(i, j-1) == 'e' || r(i, j+1) == 'e') {
                    positions.emplace_back(i, j);
                }
            }
        }
    }
    return positions;
}

// Optimized position sampling using Fisher-Yates shuffle
std::vector<std::pair<int, int>> sample_positions(std::vector<std::pair<int, int>>& positions, size_t n) {
    if (n >= positions.size()) {
        return positions;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Fisher-Yates shuffle
    for (size_t i = positions.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(gen);
        std::swap(positions[i], positions[j]);
    }

    positions.resize(n);
    return positions;
}

} // namespace

py::array_t<char> build_terrain(py::array_t<char> level, py::list agents, py::dict objects) {
    auto arr = level.mutable_unchecked<2>();
    int rows = arr.shape(0);
    int cols = arr.shape(1);

    // Remove existing agents in a single pass
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (arr(i, j) == 'a') { // 'a' for agent
                arr(i, j) = 'e';
            }
        }
    }

    std::vector<std::pair<int, int>> valid = get_valid_positions(level);
    auto agent_positions = sample_positions(valid, agents.size());

    // Place agents
    for (size_t i = 0; i < agents.size() && i < agent_positions.size(); ++i) {
        auto [r_pos, c_pos] = agent_positions[i];
        arr(r_pos, c_pos) = 'a';
    }

    // Calculate total objects and adjust if needed
    int area = rows * cols;
    int total_objects = agents.size();
    std::vector<std::pair<std::string, int>> object_counts;
    object_counts.reserve(objects.size());

    for (auto item : objects) {
        int count = item.second.cast<int>();
        total_objects += count;
        object_counts.emplace_back(py::str(item.first), count);
    }

    // Adjust object counts if needed
    while (total_objects > 2 * area / 3) {
        total_objects = agents.size();
        for (auto& [name, count] : object_counts) {
            count = std::max(1, count / 2);
            total_objects += count;
        }
    }

    // Place objects
    for (const auto& [name, count] : object_counts) {
        valid = get_valid_positions(level);
        auto spots = sample_positions(valid, count);
        for (auto [r_pos, c_pos] : spots) {
            arr(r_pos, c_pos) = name[0]; // Use first character as identifier
        }
    }

    return level;
}

PYBIND11_MODULE(terrain_builder, m) {
    m.doc() = "Fast terrain building utilities";
    m.def("build_terrain", &build_terrain, "Build terrain from numpy array");
}

