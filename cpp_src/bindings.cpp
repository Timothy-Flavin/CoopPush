#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // Needed for std::vector conversion
#include <pybind11/numpy.h>   // Needed for numpy array conversion
#include <pybind11/pytypes.h> // Needed for py::dict, py::tuple
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <queue>
#include <atomic>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
#include "coop_push.h"
#include "vectorized_environment.h"

namespace py = pybind11;

PYBIND11_MODULE(cooppush_cpp, m)
{
     m.doc() = "Pybind11 backend for the Cooperative Push PettingZoo environment";

     // Expose the CoopPushEnvironment class to Python
     py::class_<CoopPushEnvironment>(m, "Environment")
         .def(py::init<>()) // Expose the constructor
         .def("init", &CoopPushEnvironment::init,
              "Initializes the environment with starting positions for all entities.",
              py::arg("particle_positions"),
              py::arg("boulder_positions"),
              py::arg("landmark_positions"),
              py::arg("n_physics_steps"),
              py::arg("sparse_rewards"),
              py::arg("visit_all"),
              py::arg("sparse_weight"),
              py::arg("dt") = 0.1,
              py::arg("boulder_weight") = 5.0,
              py::arg("truncate_after_steps") = 1000)
         .def("reset", &CoopPushEnvironment::reset,
              "Resets the environment to the initial state and returns (state, observations).")
         .def("step",
              static_cast<py::tuple (CoopPushEnvironment::*)(py::dict)>(&CoopPushEnvironment::step),
              "Steps the environment with a dictionary of actions.",
              py::arg("actions"));

     py::class_<VectorizedCoopPush>(m, "VectorizedEnvironment")
         .def(py::init<>()) // Default constructor
         .def(py::init<std::vector<double>,
                       std::vector<double>,
                       std::vector<double>,
                       int,
                       bool,
                       bool,
                       double,
                       double,
                       double,
                       int,
                       int,
                       int,
                       int>(),
              "Initializes multiple environments in parallel.",
              py::arg("particle_positions"),
              py::arg("boulder_positions"),
              py::arg("landmark_positions"),
              py::arg("n_physics_steps") = 5,
              py::arg("sparse_rewards") = true,
              py::arg("visit_all") = true,
              py::arg("sparse_weight") = 5.0,
              py::arg("dt") = 0.1,
              py::arg("boulder_weight") = 5.0,
              py::arg("truncate_after_steps") = 500,
              py::arg("n_threads") = 4,
              py::arg("n_envs") = 16,
              py::arg("envs_per_job") = 1)

         .def("reset", &VectorizedCoopPush::reset,
              "Resets the parallel environment to the initial state and returns (state, observations).")
         .def("reset_i", &VectorizedCoopPush::reset_i,
              "Resets the parallel environment to the initial state and returns (state, observations).")
         .def("step", &VectorizedCoopPush::step,
              "Steps the parallel environment with a numpy array of actions shaped (num_envs, num_particles, 2).",
              py::arg("actions"));
}