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
        for (int i = 0; i < n_particles; ++i)
        {
            std::vector<double> obs_vec(n_particles * 4 + n_boulders * 2 + n_landmarks * 2, 0);
            std::string agent_id = "particle_" + std::to_string(i);

            for (int p = 0; p < n_particles; ++p)
            {
                obs_vec[p * 4] = current_particle_positions_[p * 2];
                obs_vec[p * 4 + 1] = current_particle_positions_[p * 2 + 1];
                obs_vec[p * 4 + 2] = current_particle_velocities_[p * 2];
                obs_vec[p * 4 + 3] = current_particle_velocities_[p * 2 + 1];
            }
            for (int b = 0; b < n_boulders; ++b)
            {
                obs_vec[n_particles * 4 + 2 * b] = current_boulder_positions_[2 * b];
                obs_vec[n_particles * 4 + 2 * b + 1] = current_boulder_positions_[2 * b + 1];
            }
            for (int l = 0; l < n_landmarks; ++l)
            {
                obs_vec[n_particles * 4 + 2 * n_boulders + 2 * l] = current_landmark_positions_[2 * l];
                obs_vec[n_particles * 4 + 2 * n_boulders + 2 * l + 1] = current_landmark_positions_[2 * l + 1];
            }
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
    std::vector<double> next_particle_positions_;
    std::vector<double> next_boulder_positions_;
    std::vector<double> next_landmark_positions_;
    std::vector<double> current_particle_velocities_;
    std::vector<double> current_boulder_velocities_;

    std::vector<bool> landmark_pairs;
    int n_landmarks = 0;
    int n_particles = 1;
    int n_boulders = 1;
    int n_lm = 1; // While this is more than zero the env stays on

    const double LANDMARK_R = 1.0;
    const double BOULDER_R = 5.0;
    const double PARTICLE_R = 1.0;
    const double total_radius = BOULDER_R + PARTICLE_R;
    const double total_radius_sq = total_radius * total_radius;

    const double TOTAL_BOULDER_R = 2.0 * BOULDER_R;
    const double TOTAL_BOULDER_R_SQ = TOTAL_BOULDER_R * TOTAL_BOULDER_R;

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

        next_particle_positions_ = particle_positions;
        next_boulder_positions_ = boulder_positions;
        next_landmark_positions_ = landmark_positions;

        n_landmarks = static_cast<int>(initial_landmark_positions_.size() / 2);
        n_boulders = static_cast<int>(initial_landmark_positions_.size() / 2);
        n_particles = static_cast<int>(initial_particle_positions_.size() / 2);

        current_boulder_velocities_.resize(initial_boulder_positions_.size(), 0.0);
        current_particle_velocities_.resize(initial_particle_positions_.size(), 0.0);
        std::fill(current_boulder_velocities_.begin(), current_boulder_velocities_.end(), 0);
        std::fill(current_particle_velocities_.begin(), current_particle_velocities_.end(), 0);

        landmark_pairs.resize(n_landmarks * n_boulders, false);
        std::fill(landmark_pairs.begin(), landmark_pairs.end(), false);
        num_particles_ = static_cast<int>(particle_positions.size() / 2); // Assuming 2D (x, y)
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
        next_particle_positions_ = initial_particle_positions_;
        next_boulder_positions_ = initial_boulder_positions_;
        next_landmark_positions_ = initial_landmark_positions_;
        current_boulder_velocities_.resize(initial_boulder_positions_.size(), 0.0);
        current_particle_velocities_.resize(initial_particle_positions_.size(), 0.0);
        std::fill(current_boulder_velocities_.begin(), current_boulder_velocities_.end(), 0);
        std::fill(current_particle_velocities_.begin(), current_particle_velocities_.end(), 0);
        // Prepare return values
        py::array_t<double> global_state = get_global_state();
        py::dict observations = get_observations();

        return py::make_tuple(global_state, observations);
    }

    void set_naive_next_pos(py::dict &actions)
    {
        for (auto item : actions)
        {
            std::string agent_id = py::str(item.first);
            // Assuming continuous actions as a numpy array [dx, dy]
            py::array_t<double> action_array = py::cast<py::array_t<double>>(item.second);
            auto buf = action_array.request();
            double *ptr = static_cast<double *>(buf.ptr);

            // Example: "particle_0" -> index 0
            int particle_index = std::stoi(agent_id.substr(agent_id.find("_") + 1));
            if (ptr[0] > 1.0)
                ptr[0] = 1.0;
            if (ptr[0] < -1.0)
                ptr[0] = -1.0;
            if (ptr[1] > 1.0)
                ptr[1] = 1.0;
            if (ptr[1] < -1.0)
                ptr[1] = -1.0;

            double mag = std::sqrt(ptr[0] * ptr[0] + ptr[1] * ptr[1]);
            if (mag < 1.0e-8)
            {
                mag = 1.0e-8;
            }
            ptr[0] /= mag;
            ptr[1] /= mag;

            if (particle_index < num_particles_)
            {
                next_particle_positions_[particle_index * 2] += ptr[0] * 0.1;     // Update x
                next_particle_positions_[particle_index * 2 + 1] += ptr[1] * 0.1; // Update y
            }
        }
    }

    void move_things()
    {
        std::vector<double> boulder_displacements(n_boulders * 2, 0.0);

        // add overlap from particles to boulder displacements
        for (int b = 0; b < n_boulders; ++b)
        {
            double b_move_x = 0.0;
            double b_move_y = 0.0;
            for (int p = 0; p < n_particles; ++p)
            {
                double dx = current_boulder_positions_[b * 2] - next_particle_positions_[p * 2];
                double dy = current_boulder_positions_[b * 2 + 1] - next_particle_positions_[p * 2 + 1];
                double particle_dist_sqr = dx * dx + dy * dy;
                if (particle_dist_sqr < total_radius_sq)
                {
                    double d = std::sqrt(particle_dist_sqr);
                    double overlap_dist = total_radius - d;

                    double move_dist = (1.0 / 10.0) * overlap_dist;

                    double normal_x = dx / d;
                    double normal_y = dy / d;

                    b_move_x += normal_x * move_dist;
                    b_move_y += normal_y * move_dist;
                }
            }
            boulder_displacements[b * 2] += b_move_x;
            boulder_displacements[b * 2 + 1] += b_move_y;
        }

        // add overlap from other boulders to boulder displacements
        for (int i = 0; i < n_boulders; ++i)
        {
            // Start the inner loop from i + 1 to avoid self-collision and duplicate checks
            for (int j = i + 1; j < n_boulders; ++j)
            {
                // Calculate the vector from boulder i to boulder j
                double dx = current_boulder_positions_[j * 2] - current_boulder_positions_[i * 2];
                double dy = current_boulder_positions_[j * 2 + 1] - current_boulder_positions_[i * 2 + 1];
                double dist_sqr = dx * dx + dy * dy;
                if (dist_sqr < total_radius_sq)
                {
                    double d = std::sqrt(dist_sqr);
                    double overlap_dist = total_radius - d;
                    double move_dist = (1.0 / 2.0) * overlap_dist;

                    // Normalize the vector from i to j to get the direction
                    double normal_x = dx / d;
                    double normal_y = dy / d;

                    boulder_displacements[i * 2] -= normal_x * move_dist;
                    boulder_displacements[i * 2 + 1] -= normal_y * move_dist;

                    boulder_displacements[j * 2] += normal_x * move_dist;
                    boulder_displacements[j * 2 + 1] += normal_y * move_dist;
                }
            }
        }

        // move boulders
        for (int b = 0; b < n_boulders; ++b)
        {
            current_boulder_positions_[b * 2] += boulder_displacements[b * 2];
            current_boulder_positions_[b * 2 + 1] += boulder_displacements[b * 2 + 1];
        }

        std::vector<double> particle_displacements(n_particles * 2, 0.0);
        // Iterate through each particle
        for (int p = 0; p < n_particles; ++p)
        {
            // Iterate through each boulder to check for a collision
            for (int b = 0; b < n_boulders; ++b)
            {
                // Calculate the vector from the boulder to the particle
                double dx = next_particle_positions_[p * 2] - current_boulder_positions_[b * 2];
                double dy = next_particle_positions_[p * 2 + 1] - current_boulder_positions_[b * 2 + 1];

                // Calculate the squared distance
                double dist_sqr = dx * dx + dy * dy;

                // Check for overlap
                if (dist_sqr < total_radius_sq)
                {
                    double d = std::sqrt(dist_sqr);

                    // Calculate the overlap distance
                    double overlap_dist = total_radius - d;

                    // Normalize the vector from the boulder to the particle
                    double normal_x = dx / d;
                    double normal_y = dy / d;

                    // Add the displacement to the particle's total displacement vector
                    particle_displacements[p * 2] += normal_x * overlap_dist;
                    particle_displacements[p * 2 + 1] += normal_y * overlap_dist;
                }
            }
        }
        for (int p = 0; p < n_particles; ++p)
        {
            current_particle_positions_[p * 2] = next_particle_positions_[p * 2] + particle_displacements[p * 2];
            current_particle_positions_[p * 2 + 1] = next_particle_positions_[p * 2 + 1] + particle_displacements[p * 2 + 1];
        }
    }

    double update_landmarks()
    {
        double r = 0.0;
        for (int l = 0; l < n_landmarks; ++l)
        {
            for (int b = 0; b < n_boulders; ++b)
            {
                double dx = current_boulder_positions_[b * 2] - current_landmark_positions_[l * 2];
                double dy = current_boulder_positions_[b * 2 + 1] - current_landmark_positions_[l * 2 + 1];
                if (!landmark_pairs[l * n_boulders + b] && total_radius_sq < dx * dx + dy * dy)
                {
                    landmark_pairs[l * n_boulders + b] = true;
                    r += 1.0;
                }
                n_lm += landmark_pairs[l * n_boulders + b];
            }
        }
        return r;
    }
    // Steps the simulation forward based on agent actions.
    py::tuple step(py::dict actions)
    {
        std::cout << "C++ step() called." << std::endl;

        // Sets the particle's desired locations (normalized from action directions)
        set_naive_next_pos(actions);

        // After handling boulder collisions / movement, sets the particles actual locations
        move_things();

        // Reward of 1 for each boulder that collides with a landmark for the first time
        double r = update_landmarks();

        py::array_t<double> global_state = get_global_state();
        py::dict observations = get_observations();
        py::dict rewards = py::dict();
        py::dict terminations = py::dict();
        py::dict truncations = py::dict();

        for (int i = 0; i < num_particles_; ++i)
        {
            std::string agent_id = "particle_" + std::to_string(i);
            rewards[py::str(agent_id)] = r;
            terminations[py::str(agent_id)] = n_lm == 0;
            truncations[py::str(agent_id)] = false;
        }

        return py::make_tuple(global_state, observations, rewards, terminations, truncations);
    }
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