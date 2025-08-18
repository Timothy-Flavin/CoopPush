#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // Needed for std::vector conversion
#include <pybind11/numpy.h>   // Needed for numpy array conversion
#include <pybind11/pytypes.h> // Needed for py::dict, py::tuple
#include <vector>
#include <string>
#include <iostream>

namespace py = pybind11;

class CoopPushEnvironment
{
public:
    // Constructor
    CoopPushEnvironment()
    {
        std::cout << "C++ CoopPushEnvironment constructed." << std::endl;
    }

    // Initializes or re-initializes the environment with entity positions.
    void init(
        std::vector<double> particle_positions,
        std::vector<double> boulder_positions,
        std::vector<double> landmark_positions)
    {
        std::cout << "C++ init() called." << std::endl;
        // Store the initial state so we can reset to it later
        initial_particle_positions_ = particle_positions;
        initial_boulder_positions_ = boulder_positions;
        initial_landmark_positions_ = landmark_positions;

        // Set the current state from the initial state
        current_particle_positions_ = particle_positions;
        current_boulder_positions_ = boulder_positions;
        current_landmark_positions_ = landmark_positions;

        num_particles_ = particle_positions.size() / 2; // Assuming 2D (x, y)
    }

    // Resets the environment to the last initialized state.
    py::tuple reset()
    {
        std::cout << "C++ reset() called." << std::endl;
        if (initial_particle_positions_.empty())
        {
            throw std::runtime_error("Environment must be initialized with init() before reset() can be called.");
        }

        // Reset current state to the stored initial state
        current_particle_positions_ = initial_particle_positions_;
        current_boulder_positions_ = initial_boulder_positions_;
        current_landmark_positions_ = initial_landmark_positions_;

        // Prepare return values
        py::array_t<double> global_state = get_global_state();
        py::dict observations = get_observations();

        return py::make_tuple(global_state, observations);
    }

    // Steps the simulation forward based on agent actions.
    py::tuple step(py::dict actions)
    {
        std::cout << "C++ step() called." << std::endl;

        // --- 1. Process Actions and Update State (Your core logic goes here) ---
        // This is a placeholder. You would implement your physics and game logic here.
        for (auto item : actions)
        {
            std::string agent_id = py::str(item.first);
            // Assuming continuous actions as a numpy array [dx, dy]
            py::array_t<double> action_array = py::cast<py::array_t<double>>(item.second);
            auto buf = action_array.request();
            double *ptr = static_cast<double *>(buf.ptr);

            // Example: "particle_0" -> index 0
            int particle_index = std::stoi(agent_id.substr(agent_id.find("_") + 1));

            if (particle_index < num_particles_)
            {
                current_particle_positions_[particle_index * 2] += ptr[0] * 0.1;     // Update x
                current_particle_positions_[particle_index * 2 + 1] += ptr[1] * 0.1; // Update y
            }
        }

        // --- 2. Prepare Return Values in Python format ---
        py::array_t<double> global_state = get_global_state();
        py::dict observations = get_observations();
        py::dict rewards = py::dict();
        py::dict terminations = py::dict();
        py::dict truncations = py::dict();

        bool is_done = false; // Example termination condition
        if (current_particle_positions_[0] > 1.0)
        { // particle_0 goes out of bounds
            is_done = true;
        }

        for (int i = 0; i < num_particles_; ++i)
        {
            std::string agent_id = "particle_" + std::to_string(i);
            // Example reward: negative distance from origin for each particle
            double x = current_particle_positions_[i * 2];
            double y = current_particle_positions_[i * 2 + 1];
            rewards[py::str(agent_id)] = -std::sqrt(x * x + y * y);
            terminations[py::str(agent_id)] = is_done;
            truncations[py::str(agent_id)] = false;
        }

        return py::make_tuple(global_state, observations, rewards, terminations, truncations);
    }

private:
    // Helper function to get the full global state as a NumPy array.
    py::array_t<double> get_global_state()
    {
        std::vector<double> state_vec;
        state_vec.insert(state_vec.end(), current_particle_positions_.begin(), current_particle_positions_.end());
        state_vec.insert(state_vec.end(), current_boulder_positions_.begin(), current_boulder_positions_.end());
        state_vec.insert(state_vec.end(), current_landmark_positions_.begin(), current_landmark_positions_.end());

        return py::array_t<double>(state_vec.size(), state_vec.data());
    }

    // Helper function to get agent-specific observations as a Python dictionary.
    py::dict get_observations()
    {
        py::dict obs_dict;
        for (int i = 0; i < num_particles_; ++i)
        {
            std::string agent_id = "particle_" + std::to_string(i);
            // Example observation: each agent sees its own position
            std::vector<double> obs_vec = {
                current_particle_positions_[i * 2],
                current_particle_positions_[i * 2 + 1]};
            obs_dict[py::str(agent_id)] = py::array_t<double>(obs_vec.size(), obs_vec.data());
        }
        return obs_dict;
    }

    // --- Member Variables ---
    int num_particles_ = 0;
    // Initial state (used for reset)
    std::vector<double> initial_particle_positions_;
    std::vector<double> initial_boulder_positions_;
    std::vector<double> initial_landmark_positions_;
    // Current state
    std::vector<double> current_particle_positions_;
    std::vector<double> current_boulder_positions_;
    std::vector<double> current_landmark_positions_;
};

// --- PYBIND11 MODULE DEFINITION ---
// The first argument is the name of the module (e.g., import cooppush_cpp)
// The second argument, 'm', is a variable of type py::module_ which is the main interface
PYBIND11_MODULE(cooppush_cpp, m)
{
    m.doc() = "Pybind11 backend for the Cooperative Push PettingZoo environment";

    // Expose the CoopPushEnvironment class to Python
    py::class_<CoopPushEnvironment>(m, "Environment")
        .def(py::init<>()) // Expose the constructor
        .def("init", &CoopPushEnvironment::init,
             "Initializes the environment with starting positions for all entities.",
             py::arg("particle_positions"), py::arg("boulder_positions"), py::arg("landmark_positions"))
        .def("reset", &CoopPushEnvironment::reset,
             "Resets the environment to the initial state and returns (state, observations).")
        .def("step", &CoopPushEnvironment::step,
             "Steps the environment with a dictionary of actions.",
             py::arg("actions"));
}