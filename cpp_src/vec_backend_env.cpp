#include "vec_backend_env.h"

VecBackendEnv::VecBackendEnv(std::vector<double> particle_positions,
                             std::vector<double> boulder_positions,
                             std::vector<double> landmark_positions,
                             int n_physics_steps,
                             bool sparse_rewards,
                             bool visit_all,
                             double sparse_weight,
                             double dt,
                             double boulder_weight,
                             int truncate_after_steps)
{
    int visit_every_state_size = 0;
    if (visit_all)
        visit_every_state_size = n_boulders * n_landmarks;
    else
        visit_every_state_size = n_boulders;
    this->global_state_size = n_particles * 4 + n_boulders * 2 + n_landmarks * 2 + visit_every_state_size;
}

StepResult VecBackendEnv::step(const double *actions)
{
    ++current_step;
    double r = 0.0;
    for (int i = 0; i < n_physics_steps_; ++i)
    {
        set_naive_next_pos(actions);
        move_things();
        if (visit_all)
            r += get_reward_all();
        else
            r += get_reward_one();
    }
    std::vector<double> global_state = _get_global_state();

    bool term = false;
    if (visit_all)
        term = n_lm == 0;
    else
        term = n_active_boulders == 0;
    bool trunc = current_step >= truncate_after_steps_;
    return std::make_tuple(global_state, r, term, trunc);
}

std::vector<double> VecBackendEnv::reset()
{
    // Reset current state to the stored initial state
    this->current_step = 0;
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
    std::vector<double> global_state = _get_global_state();
    landmark_pairs.resize(n_landmarks * n_boulders, false);
    std::fill(landmark_pairs.begin(), landmark_pairs.end(), false);
    finished_boulders.resize(n_boulders, false);
    std::fill(finished_boulders.begin(), finished_boulders.end(), false);

    n_lm = 1;
    n_active_boulders = 1;

    return global_state;
}
// Public helper to access current global state as a plain vector (no NumPy types)
std::vector<double> VecBackendEnv::get_state_vector()
{
    return _get_global_state();
}

std::vector<double> VecBackendEnv::_get_global_state()
{
    int visit_every_state_size = 0;
    if (visit_all)
        visit_every_state_size = n_boulders * n_landmarks;
    else
        visit_every_state_size = n_boulders;

    // std::cout << "npart4: " << n_particles * 4 << " nb*2 " << n_boulders * 2 << " nlandmark2 " << n_landmarks * 2 << " vess " << visit_every_state_size << std::endl;
    std::vector<double> state_vec(n_particles * 4 + n_boulders * 2 + n_landmarks * 2 + visit_every_state_size, 0);

    for (int p = 0; p < n_particles; ++p)
    {
        state_vec[p * 4] = current_particle_positions_[p * 2];
        state_vec[p * 4 + 1] = current_particle_positions_[p * 2 + 1];
        state_vec[p * 4 + 2] = current_particle_velocities_[p * 2];
        state_vec[p * 4 + 3] = current_particle_velocities_[p * 2 + 1];
    }
    for (int b = 0; b < n_boulders; ++b)
    {
        state_vec[n_particles * 4 + 2 * b] = current_boulder_positions_[2 * b];
        state_vec[n_particles * 4 + 2 * b + 1] = current_boulder_positions_[2 * b + 1];
    }
    for (int l = 0; l < n_landmarks; ++l)
    {
        state_vec[n_particles * 4 + 2 * n_boulders + 2 * l] = current_landmark_positions_[2 * l];
        state_vec[n_particles * 4 + 2 * n_boulders + 2 * l + 1] = current_landmark_positions_[2 * l + 1];
    }
    if (visit_all)
    {
        for (int v = 0; v < visit_every_state_size; ++v)
        {
            state_vec[n_particles * 4 + 2 * n_boulders + 2 * n_landmarks + v] = landmark_pairs[v];
        }
    }
    else
    {
        for (int v = 0; v < visit_every_state_size; ++v)
        {
            state_vec[n_particles * 4 + 2 * n_boulders + 2 * n_landmarks + v] = finished_boulders[v];
        }
    }
    return state_vec;
}

int VecBackendEnv::state_size()
{
    return this->global_state_size;
}