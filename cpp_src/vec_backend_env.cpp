#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for std::vector conversion
#include <pybind11/numpy.h>
#include "vec_backend_env.h"

void VecBackendEnv::set_naive_next_pos(const double *actions)
{
    // make a copy of actions
    std::vector<double> action_vec(actions, actions + n_particles * 2);
    for (int ai = 0; ai < this->n_particles; ++ai)
    {
        if (action_vec[ai * 2] > 1.0)
            action_vec[ai * 2] = 1.0;
        if (action_vec[ai * 2] < -1.0)
            action_vec[ai * 2] = -1.0;
        if (action_vec[ai * 2 + 1] > 1.0)
            action_vec[ai * 2 + 1] = 1.0;
        if (action_vec[ai * 2 + 1] < -1.0)
            action_vec[ai * 2 + 1] = -1.0;

        double mag = std::sqrt(action_vec[ai * 2] * action_vec[ai * 2] + action_vec[ai * 2 + 1] * action_vec[ai * 2 + 1]);
        if (mag < 1.0)
        {
            mag = 1.0;
        }
        action_vec[ai * 2] /= mag;
        action_vec[ai * 2 + 1] /= mag;

        current_particle_velocities_[ai * 2] = action_vec[ai * 2];
        current_particle_velocities_[ai * 2 + 1] = action_vec[ai * 2 + 1];
        next_particle_positions_[ai * 2] = current_particle_positions_[ai * 2] + action_vec[ai * 2] * delta_time;             // Update x
        next_particle_positions_[ai * 2 + 1] = current_particle_positions_[ai * 2 + 1] + action_vec[ai * 2 + 1] * delta_time; // Update y
    }
}
void VecBackendEnv::move_things()
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

                double move_dist = (1.0 / boulder_weight) * overlap_dist;

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
            if (dist_sqr < TOTAL_BOULDER_R_SQ)
            {
                double d = std::sqrt(dist_sqr);
                double overlap_dist = TOTAL_BOULDER_R - d;
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

    if (!sparse_rewards)
    {
        displacement_reward = 0.0;
        for (int b = 0; b < n_boulders; ++b)
        {
            int cl = -1;
            double cd = 10000000.0;
            double bdx = 1.0e-8;
            double bdy = 1.0e-8;
            for (int l = 0; l < n_landmarks; ++l)
            {
                double dx = current_boulder_positions_[b * 2] - current_landmark_positions_[l * 2];
                double dy = current_boulder_positions_[b * 2 + 1] - current_landmark_positions_[l * 2 + 1];
                if (dx * dx + dy * dy < cd && !landmark_pairs[l * n_boulders + b] && !finished_boulders[b])
                {
                    // std::cout << "Boulder " << b << " closest to landmark " << l << " dx: " << dx << " dy: " << dy << std::endl;
                    cd = dx * dx + dy * dy;
                    cl = l;
                    bdx = dx;
                    bdy = dy;
                }
            }

            if (cl != -1)
            {
                // double dist = std::sqrt(bdx * bdx + bdy * bdy);
                bdx += boulder_displacements[b * 2];
                bdy += boulder_displacements[b * 2 + 1];
                // std::cout << "Boulder lowest dx dy" << b << " closest to landmark " << cl << " bdx: " << bdx << " bdy: " << bdy << std::endl;
                displacement_reward -= std::sqrt(bdx * bdx + bdy * bdy) - std::sqrt(cd);
                // std::cout << "Distances " << std::sqrt(bdx * bdx + bdy * bdy) << " , " << std::sqrt(cd) << " displacement reward: " << displacement_reward << std::endl;
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

                // std::cout << "Particle position: " << next_particle_positions_[p * 2] << "," << next_particle_positions_[p * 2 + 1] << ", boulder position: " << current_boulder_positions_[b * 2] << "," << current_boulder_positions_[b * 2 + 1] << ", dx: " << normal_x * overlap_dist << ", dy: " << normal_y * overlap_dist << std::endl;

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
        next_particle_positions_[p * 2] = current_particle_positions_[p * 2];
        next_particle_positions_[p * 2 + 1] = current_particle_positions_[p * 2 + 1];
    }
}

double VecBackendEnv::get_reward_all()
{
    double r = 0.0;
    n_lm = 0;
    n_active_boulders = 1;
    for (int l = 0; l < n_landmarks; ++l)
    {
        for (int b = 0; b < n_boulders; ++b)
        {
            double dx = current_boulder_positions_[b * 2] - current_landmark_positions_[l * 2];
            double dy = current_boulder_positions_[b * 2 + 1] - current_landmark_positions_[l * 2 + 1];
            if (!landmark_pairs[l * n_boulders + b] && total_radius_sq > dx * dx + dy * dy)
            {
                landmark_pairs[l * n_boulders + b] = true;
                r += sparse_weight;
            }
            n_lm += !landmark_pairs[l * n_boulders + b];
        }
    }
    if (!sparse_rewards)
        r += displacement_reward;
    return r;
}
double VecBackendEnv::get_reward_one()
{
    double r = 0.0;
    n_active_boulders = n_boulders;
    n_lm = 1;

    for (int b = 0; b < n_boulders; ++b)
    {
        if (finished_boulders[b])
        {
            --n_active_boulders;
            continue;
        }
        for (int l = 0; l < n_landmarks; ++l)
        {
            double dx = current_boulder_positions_[b * 2] - current_landmark_positions_[l * 2];
            double dy = current_boulder_positions_[b * 2 + 1] - current_landmark_positions_[l * 2 + 1];
            if (total_radius_sq > dx * dx + dy * dy)
            {
                landmark_pairs[l * n_boulders + b] = true;
                finished_boulders[b] = true;
                --n_active_boulders;
                r += 1.0;
                break;
            }
        }
    }
    if (!sparse_rewards)
        r += displacement_reward;
    return r;
}

VecBackendEnv::VecBackendEnv() : my_index(0)
{
}
VecBackendEnv::VecBackendEnv(std::vector<double> particle_positions,
                             std::vector<double> boulder_positions,
                             std::vector<double> landmark_positions,
                             int n_physics_steps,
                             bool sparse_rewards,
                             bool visit_all,
                             double sparse_weight,
                             double dt,
                             double boulder_weight,
                             int truncate_after_steps,
                             const int idx) : my_index(idx)
{
    int visit_every_state_size = 0;
    if (visit_all)
        visit_every_state_size = n_boulders * n_landmarks;
    else
        visit_every_state_size = n_boulders;
    this->global_state_size = n_particles * 4 + n_boulders * 2 + n_landmarks * 2 + visit_every_state_size;
    // std::cout << "C++ init() called." << std::endl;
    this->n_physics_steps_ = n_physics_steps;
    this->truncate_after_steps_ = truncate_after_steps;
    this->current_step = 0;
    delta_time = dt;
    this->visit_all = visit_all;
    this->sparse_rewards = sparse_rewards;
    this->sparse_weight = sparse_weight;
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
    n_boulders = static_cast<int>(initial_boulder_positions_.size() / 2);
    n_particles = static_cast<int>(initial_particle_positions_.size() / 2);

    current_boulder_velocities_.resize(initial_boulder_positions_.size(), 0.0);
    current_particle_velocities_.resize(initial_particle_positions_.size(), 0.0);
    std::fill(current_boulder_velocities_.begin(), current_boulder_velocities_.end(), 0);
    std::fill(current_particle_velocities_.begin(), current_particle_velocities_.end(), 0);

    landmark_pairs.resize(n_landmarks * n_boulders, false);
    std::fill(landmark_pairs.begin(), landmark_pairs.end(), false);
    finished_boulders.resize(n_boulders, false);
    std::fill(finished_boulders.begin(), finished_boulders.end(), false);
    num_particles_ = static_cast<int>(particle_positions.size() / 2);
}

void VecBackendEnv::step(const double *actions, double *obs_ptr, double *rewards_ptr, bool *terminateds_ptr, bool *truncateds_ptr)
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
    // std::vector<double> global_state = get_global_state();

    bool term = false;
    if (visit_all)
        term = n_lm == 0;
    else
        term = n_active_boulders == 0;
    bool trunc = current_step >= truncate_after_steps_;

    get_global_state(obs_ptr);
    rewards_ptr[my_index * sizeof(double)] = r;
    terminateds_ptr[my_index * sizeof(bool)] = term;
    truncateds_ptr[my_index * sizeof(bool)] = trunc;
    // TODO: actually copy the results in
    //  return StepResult{global_state, r, term, trunc};
}

void VecBackendEnv::reset(double *global_state_ptr)
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
    landmark_pairs.resize(n_landmarks * n_boulders, false);
    std::fill(landmark_pairs.begin(), landmark_pairs.end(), false);
    finished_boulders.resize(n_boulders, false);
    std::fill(finished_boulders.begin(), finished_boulders.end(), false);
    n_lm = 1;
    n_active_boulders = 1;
    get_global_state(global_state_ptr);

    // std::cout << "  reset successful" << std::endl;
}
// Public helper to access current global state as a plain vector (no NumPy types)

void VecBackendEnv::get_global_state(double *global_state_ptr)
{
    int visit_every_state_size = 0;
    if (this->visit_all)
        visit_every_state_size = n_boulders * n_landmarks;
    else
        visit_every_state_size = n_boulders;

    // std::cout << "    npart4: " << n_particles * 4 << " nb*2 " << n_boulders * 2 << " nlandmark2 " << n_landmarks * 2 << " vess " << visit_every_state_size << std::endl;
    // std::vector<double> state_vec(n_particles * 4 + n_boulders * 2 + n_landmarks * 2 + visit_every_state_size, 0);
    double *state_vec = global_state_ptr + this->global_state_size * my_index * sizeof(double);

    for (int p = 0; p < n_particles; ++p)
    {
        state_vec[p * 4] = current_particle_positions_[p * 2];
        state_vec[p * 4 + 1] = current_particle_positions_[p * 2 + 1];
        state_vec[p * 4 + 2] = current_particle_velocities_[p * 2];
        state_vec[p * 4 + 3] = current_particle_velocities_[p * 2 + 1];
    }
    // std::cout << "    got particles" << std::endl;
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
    // return state_vec;
}

int VecBackendEnv::state_size()
{
    return this->global_state_size;
}