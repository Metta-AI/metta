#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<std::pair<int, int>> get_valid_positions(py::array level) {
    int rows = level.shape(0);
    int cols = level.shape(1);
    std::vector<std::pair<int, int>> positions;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            std::string cell = py::str(level.attr("__getitem__")(py::make_tuple(i, j)));
            if (cell == "empty") {
                if (
                    std::string(py::str(level.attr("__getitem__")(py::make_tuple(i - 1, j)))) == "empty" ||
                    std::string(py::str(level.attr("__getitem__")(py::make_tuple(i + 1, j)))) == "empty" ||
                    std::string(py::str(level.attr("__getitem__")(py::make_tuple(i, j - 1)))) == "empty" ||
                    std::string(py::str(level.attr("__getitem__")(py::make_tuple(i, j + 1)))) == "empty") {
                    positions.emplace_back(i, j);
                }
            }
        }
    }
    return positions;
}

// Sample n unique positions from positions vector
std::vector<std::pair<int, int>> sample_positions(std::vector<std::pair<int, int>> positions, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(positions.begin(), positions.end(), gen);
    if (n > positions.size()) n = positions.size();
    return std::vector<std::pair<int, int>>(positions.begin(), positions.begin() + n);
}

} // namespace

py::array build_terrain(py::array level, py::list agents, py::dict objects) {
    int rows = level.shape(0);
    int cols = level.shape(1);

    // Remove existing agents
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (std::string(py::str(level.attr("__getitem__")(py::make_tuple(r, c)))) == "agent.agent") {
                level.attr("__setitem__")(py::make_tuple(r, c), py::str("empty"));
            }
        }
    }

    std::vector<std::pair<int, int>> valid = get_valid_positions(level);
    auto agent_positions = sample_positions(valid, agents.size());
    for (size_t i = 0; i < agents.size() && i < agent_positions.size(); ++i) {
        auto [r, c] = agent_positions[i];
        level.attr("__setitem__")(py::make_tuple(r, c), agents[i]);
    }

    int area = rows * cols;
    int total_objects = agents.size();
    for (auto item : objects) {
        total_objects += item.second.cast<int>();
    }
    while (total_objects > 2 * area / 3) {
        total_objects = agents.size();
        for (auto item : objects) {
            int count = item.second.cast<int>();
            int new_count = std::max(1, count / 2);
            objects[item.first] = py::int_(new_count);
            total_objects += new_count;
        }
    }

    for (auto item : objects) {
        std::string name = py::str(item.first);
        int count = item.second.cast<int>();
        valid = get_valid_positions(level);
        auto spots = sample_positions(valid, count);
        for (auto [r, c] : spots) {
            level.attr("__setitem__")(py::make_tuple(r, c), py::str(name));
        }
    }

    return level;
}

PYBIND11_MODULE(terrain_builder, m) {
    m.doc() = "Fast terrain building utilities";
    m.def("build_terrain", &build_terrain, "Build terrain from numpy array");
}

