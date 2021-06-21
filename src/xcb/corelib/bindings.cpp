#include <pybind11/pybind11.h>
#include "toy.hpp"
#include "xcb_inference.hpp"

namespace py = pybind11;

PYBIND11_PLUGIN(core)
{
    py::module m("core");
    m.def("add", &add);
    m.def("subtract", &subtract);
    py::class_<ModelChain> ModelChain(m, "ModelChain");
    ModelChain
        .def(py::init<>())
        .def("add_elements", &ModelChain::add_elements)
        .def("get_size", &ModelChain::get_size)
        .def("get_labels", &ModelChain::get_labels)
        .def("get_top_labels", &ModelChain::get_top_labels)
        .def("beam_search", &ModelChain::beam_search);
    return m.ptr();
}