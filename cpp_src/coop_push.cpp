#include "coop_push.h"

namespace py = pybind11;

// Helper function to get the full global state as a NumPy array.
py::array_t<double> CoopPushEnvironment::get_global_state()
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
    // Update member copy for observations
    global_state_vec = state_vec;

    // Return an owning NumPy array (copy) to avoid dangling pointers
    py::array_t<double> arr(state_vec.size());
    auto info = arr.request();
    std::memcpy(info.ptr, state_vec.data(), state_vec.size() * sizeof(double));
    return arr;
}
// Helper function to get agent-specific observations as a Python dictionary.
py::dict CoopPushEnvironment::get_observations()
{
    py::dict obs_dict;
    for (int i = 0; i < n_particles; ++i)
    {
        std::string agent_id = "particle_" + std::to_string(i);
        std::vector<double> obs_vec = global_state_vec; // is this a copy?
        for (int o = 0; o < 4; ++o)
            std::swap(obs_vec[i * 4 + o], obs_vec[0 + o]);
        // std::cout << "Obs vec in cpp " << obs_vec.size() << std::endl;
        py::array_t<double> arr(obs_vec.size());
        auto info = arr.request();
        std::memcpy(info.ptr, obs_vec.data(), obs_vec.size() * sizeof(double));
        obs_dict[py::str(agent_id)] = arr;
    }
    return obs_dict;
}

// Initializes or re-initializes the environment with entity positions.
void CoopPushEnvironment::init(
    std::vector<double> particle_positions,
    std::vector<double> boulder_positions,
    std::vector<double> landmark_positions,
    int n_physics_steps = 5,
    bool sparse_rewards = true,
    bool visit_all = true,
    double sparse_weight = 5.0,
    double dt = 0.1,
    double boulder_weight = 5.0,
    int truncate_after_steps = 1000)
{
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
    num_particles_ = static_cast<int>(particle_positions.size() / 2); // Assuming 2D (x, y)
}

// Resets the environment to the last initialized state.
py::tuple CoopPushEnvironment::reset()
{
    // std::cout << "C++ reset() called." << std::endl;
    if (initial_particle_positions_.empty())
    {
        throw std::runtime_error("Environment must be initialized with init() before reset() can be called.");
    }

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
    py::array_t<double> global_state = get_global_state();
    py::dict observations = get_observations();
    landmark_pairs.resize(n_landmarks * n_boulders, false);
    std::fill(landmark_pairs.begin(), landmark_pairs.end(), false);
    finished_boulders.resize(n_boulders, false);
    std::fill(finished_boulders.begin(), finished_boulders.end(), false);

    n_lm = 1;
    n_active_boulders = 1;

    return py::make_tuple(global_state, observations);
}

// Steps the environment given a dictionary of actions for each agent.
void CoopPushEnvironment::set_naive_next_pos(py::dict &actions)
{
    for (auto item : actions)
    {
        std::string agent_id = py::str(item.first);
        // Assuming continuous actions as a numpy array [dx, dy]
        py::array_t<double> action_array = py::cast<py::array_t<double>>(item.second);
        auto buf = action_array.request();
        double *ptr = static_cast<double *>(buf.ptr);

        // std::cout << agent_id << " before: " << ptr[0] << ", " << ptr[1] << " after: ";
        //  Example: "particle_0" -> index 0
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
        if (mag < 1.0)
        {
            mag = 1.0;
        }
        ptr[0] /= mag;
        ptr[1] /= mag;

        // std::cout << ptr[0] << ", " << ptr[1] << std::endl;

        if (particle_index < num_particles_)
        {
            current_particle_velocities_[particle_index * 2] = ptr[0];
            current_particle_velocities_[particle_index * 2 + 1] = ptr[1];
            next_particle_positions_[particle_index * 2] = current_particle_positions_[particle_index * 2] + ptr[0] * delta_time;         // Update x
            next_particle_positions_[particle_index * 2 + 1] = current_particle_positions_[particle_index * 2 + 1] + ptr[1] * delta_time; // Update y
        }
    }
}

// actions in the shape (n_particles, 2)
void CoopPushEnvironment::set_naive_next_pos(py::array_t<double> actions)
{
    // Accept either a (n_particles,2) array or a flat array of length n_particles*2
    auto buf = actions.request();
    if (buf.ndim == 2)
    {
        ssize_t rows = buf.shape[0];
        ssize_t cols = buf.shape[1];
        double *ptr = static_cast<double *>(buf.ptr);
        if (cols != 2)
            throw std::runtime_error("actions must have shape (n,2)");

        ssize_t use_n = std::min<ssize_t>(rows, num_particles_);
        for (ssize_t i = 0; i < use_n; ++i)
        {
            double ax = ptr[i * 2];
            double ay = ptr[i * 2 + 1];

            // clamp
            if (ax > 1.0)
                ax = 1.0;
            if (ax < -1.0)
                ax = -1.0;
            if (ay > 1.0)
                ay = 1.0;
            if (ay < -1.0)
                ay = -1.0;

            double mag = std::sqrt(ax * ax + ay * ay);
            if (mag < 1.0)
                mag = 1.0;
            ax /= mag;
            ay /= mag;

            current_particle_velocities_[i * 2] = ax;
            current_particle_velocities_[i * 2 + 1] = ay;
            next_particle_positions_[i * 2] = current_particle_positions_[i * 2] + ax * delta_time;
            next_particle_positions_[i * 2 + 1] = current_particle_positions_[i * 2 + 1] + ay * delta_time;
        }
    }
    else if (buf.ndim == 1)
    {
        ssize_t len = buf.shape[0];
        if (len % 2 != 0)
            throw std::runtime_error("flat actions array length must be multiple of 2");
        ssize_t use_n = std::min<ssize_t>(len / 2, num_particles_);
        double *ptr = static_cast<double *>(buf.ptr);
        for (ssize_t i = 0; i < use_n; ++i)
        {
            double ax = ptr[i * 2];
            double ay = ptr[i * 2 + 1];

            // clamp
            if (ax > 1.0)
                ax = 1.0;
            if (ax < -1.0)
                ax = -1.0;
            if (ay > 1.0)
                ay = 1.0;
            if (ay < -1.0)
                ay = -1.0;

            double mag = std::sqrt(ax * ax + ay * ay);
            if (mag < 1.0)
                mag = 1.0;
            ax /= mag;
            ay /= mag;

            current_particle_velocities_[i * 2] = ax;
            current_particle_velocities_[i * 2 + 1] = ay;
            next_particle_positions_[i * 2] = current_particle_positions_[i * 2] + ax * delta_time;
            next_particle_positions_[i * 2 + 1] = current_particle_positions_[i * 2 + 1] + ay * delta_time;
        }
    }
    else
    {
        throw std::runtime_error("actions must be a 1D or 2D numpy array");
    }
}

void CoopPushEnvironment::move_things()
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

double CoopPushEnvironment::get_reward_all()
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
double CoopPushEnvironment::get_reward_one()
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

py::tuple CoopPushEnvironment::step(py::dict actions)
{
    // std::cout << "C++ step() called." << std::endl;
    // print_vec(current_particle_positions_);
    // print_vec(next_particle_positions_);
    //  Sets the particle's desired locations (normalized from action directions)
    double r = 0.0;
    ++current_step;

    for (int i = 0; i < n_physics_steps_; ++i)
    {
        set_naive_next_pos(actions);
        move_things();
        if (visit_all)
            r += get_reward_all();
        else
            r += get_reward_one();
    }

    // std::cout << "After next pos: \n";
    // print_vec(current_particle_positions_);
    // print_vec(next_particle_positions_);

    // After handling boulder collisions / movement, sets the particles actual locations

    // std::cout << "After move things: \n";
    // print_vec(current_particle_positions_);
    // print_vec(next_particle_positions_);
    //  Reward of 1 for each boulder that collides with a landmark for the first time

    py::array_t<double> global_state = get_global_state();
    py::dict observations = get_observations();
    py::dict rewards = py::dict();
    py::dict terminations = py::dict();
    py::dict truncations = py::dict();

    bool term = false;
    if (visit_all)
        term = n_lm == 0;
    else
        term = n_active_boulders == 0;
    for (int i = 0; i < num_particles_; ++i)
    {
        std::string agent_id = "particle_" + std::to_string(i);
        rewards[py::str(agent_id)] = r;
        terminations[py::str(agent_id)] = term;
        truncations[py::str(agent_id)] = (current_step >= truncate_after_steps_);
    }

    return py::make_tuple(global_state, observations, rewards, terminations, truncations);
}
