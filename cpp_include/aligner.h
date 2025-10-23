// aligner.h
#pragma once // Prevents multiple inclusions in the same file
#include <pybind11/numpy.h>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
namespace py = pybind11;
py::array_t<double> create_aligned_double_buffer(ssize_t num_elements);
py::array_t<double> create_aligned_double_buffer_2d(ssize_t dim1, ssize_t dim2);
py::array_t<bool> create_aligned_bool_buffer(ssize_t num_elements);