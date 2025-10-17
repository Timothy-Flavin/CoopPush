#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // Needed for std::vector conversion
#include <pybind11/numpy.h>   // Needed for numpy array conversion
#include <pybind11/pytypes.h> // Needed for py::dict, py::tuple
#include <vector>
#include <string>
#include <iostream>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif

class CoopPushEnvironment
{
private:
    // --- Member Variables ---
    int num_particles_ = 0;
    std::vector<double> global_state_vec;

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
    std::vector<bool> finished_boulders;
    int n_landmarks = 0;
    int n_particles = 1;
    int n_boulders = 1;
    int n_lm = 1; // While this is more than zero the env stays on
    int n_active_boulders = 1;
    int n_physics_steps_ = 5;

    bool sparse_rewards = true;
    bool visit_all = true;
    double displacement_reward = 0.0;
    double sparse_weight = 1.0;
    double delta_time = 0.1;
    double boulder_weight = 5.0;
    int truncate_after_steps_ = 1000;
    int current_step = 0;

    const double LANDMARK_R = 1.0;
    const double BOULDER_R = 5.0;
    const double PARTICLE_R = 1.0;
    const double total_radius = BOULDER_R + PARTICLE_R;
    const double total_radius_sq = total_radius * total_radius;

    const double TOTAL_BOULDER_R = 2.0 * BOULDER_R;
    const double TOTAL_BOULDER_R_SQ = TOTAL_BOULDER_R * TOTAL_BOULDER_R;

    py::array_t<double> get_global_state();
    py::dict get_observations();

public:
    void init(
        std::vector<double> particle_positions,
        std::vector<double> boulder_positions,
        std::vector<double> landmark_positions,
        int n_physics_steps,
        bool sparse_rewards,
        bool visit_all,
        double sparse_weight,
        double dt,
        double boulder_weight,
        int truncate_after_steps);
    py::tuple reset();
    void set_naive_next_pos(py::dict &actions);
    void set_naive_next_pos(py::array_t<double> actions);
    void move_things();
    double get_reward_all();
    double get_reward_one();
    py::tuple step(py::dict actions);
};