#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static py::list get_valid_positions(py::array level) {
    py::buffer_info info = level.request();
    if (info.ndim != 2) {
        throw std::runtime_error("level must be 2D");
    }
    auto rows = info.shape[0];
    auto cols = info.shape[1];
    auto data = static_cast<PyObject**>(info.ptr);
    auto stride_row = info.strides[0] / sizeof(PyObject*);
    auto stride_col = info.strides[1] / sizeof(PyObject*);

    py::list positions;
    for (ssize_t i = 1; i < rows - 1; ++i) {
        for (ssize_t j = 1; j < cols - 1; ++j) {
            PyObject* cell = data[i * stride_row + j * stride_col];
            std::string value = py::str(py::handle(cell));
            if (value == "empty") {
                PyObject* up = data[(i - 1) * stride_row + j * stride_col];
                PyObject* down = data[(i + 1) * stride_row + j * stride_col];
                PyObject* left = data[i * stride_row + (j - 1) * stride_col];
                PyObject* right = data[i * stride_row + (j + 1) * stride_col];
                if (std::string(py::str(py::handle(up))) == "empty" ||
                    std::string(py::str(py::handle(down))) == "empty" ||
                    std::string(py::str(py::handle(left))) == "empty" ||
                    std::string(py::str(py::handle(right))) == "empty") {
                    positions.append(py::make_tuple(i, j));
                }
            }
        }
    }
    return positions;
}

PYBIND11_MODULE(terrain_utils, m) {
    m.doc() = "Utilities for terrain generation";
    m.def("get_valid_positions", &get_valid_positions, "Return valid spawn positions");
}
