#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for std::vector conversion
#include <pybind11/numpy.h>
#include "vectorized_environment.h"
#include "aligner.cpp"
namespace py = pybind11;

py::array_t<double> vec2d_to_pyarray(const std::vector<std::vector<double>> &vec)
{

    size_t rows = vec.size();
    size_t cols = vec[0].size();
    py::array_t<double> result({rows, cols});
    auto buffer = result.mutable_unchecked<2>();
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            buffer(i, j) = vec[i][j];
        }
    }
    return result;
}

// Convert a std::vector<double> (1D) to a py::array_t<double>
py::array_t<double> vec1d_to_pyarray(const std::vector<double> &vec)
{
    // Create a 1D NumPy array with the same length as the vector
    py::array_t<double> result(vec.size());
    // Use a mutable view to write data efficiently
    auto buffer = result.mutable_unchecked<1>();
    for (size_t i = 0; i < vec.size(); ++i)
    {
        buffer(i) = vec[i];
    }
    return result;
}

VectorizedCoopPush::VectorizedCoopPush()
{
}
VectorizedCoopPush::VectorizedCoopPush(std::vector<double> particle_positions,
                                       std::vector<double> boulder_positions,
                                       std::vector<double> landmark_positions,
                                       int n_physics_steps,
                                       bool sparse_rewards,
                                       bool visit_all,
                                       double sparse_weight,
                                       double dt,
                                       double boulder_weight,
                                       int truncate_after_steps,
                                       int n_threads,
                                       int n_envs,
                                       int envs_per_job) : m_pool((ssize_t)n_threads)
{
    std::cout << "Initializing Vectorized Environment Performance Overhaul version 1.2.2" << std::endl;
    this->m_n_envs = n_envs;
    this->m_n_threads = n_threads;
    this->m_envs_per_job = envs_per_job;
    for (int i = 0; i < n_envs; ++i)
    {
        this->m_envs.push_back(VecBackendEnv(
            particle_positions,
            boulder_positions,
            landmark_positions,
            n_physics_steps,
            sparse_rewards,
            visit_all,
            sparse_weight,
            dt,
            boulder_weight,
            truncate_after_steps,
            i));
    }
    assert(!this->m_envs.empty());
    this->m_env_obs_size = this->m_envs.at(0).state_size();

    m_buffer_1 = create_aligned_double_buffer_2d(m_n_envs, m_env_obs_size);
    m_buffer_2 = create_aligned_double_buffer_2d(m_n_envs, m_env_obs_size);
    m_rewards = create_aligned_double_buffer(n_envs);
    m_terminateds = create_aligned_bool_buffer(m_n_envs);
    m_truncateds = create_aligned_bool_buffer(m_n_envs);

    m_current_obs = m_buffer_1;
    m_next_obs = m_buffer_2;
}
py::array_t<double> VectorizedCoopPush::reset()
{
    double *obs_ptr = m_next_obs.mutable_data();
    for (int i = 0; i < m_n_envs; ++i)
    {
        this->m_envs[i].reset(obs_ptr + this->m_env_obs_size * i * sizeof(double));
    }
    // Return the buffer that is now filled with the first obs
    return m_next_obs;
}

py::array_t<double> VectorizedCoopPush::reset_i(int i)
{
    double *obs_ptr = m_next_obs.mutable_data();
    this->m_envs[i].reset(obs_ptr + this->m_env_obs_size * i * sizeof(double));
    return m_next_obs;
}

void step_job(
    py::array_t<double, py::array::c_style | py::array::forcecast> &actions,
    double *obs_ptr,
    double *rewards_ptr,
    bool *terminateds_ptr,
    bool *truncateds_ptr,
    std::vector<VecBackendEnv> &envs,
    ssize_t start_i,
    ssize_t end_i,
    ssize_t obs_size)
{
    auto actions_acc = actions.unchecked<3>(); // Shape: [n_env, n_agent, act_size]
    ssize_t num_agents = envs[0].get_num_particles();
    ssize_t action_size = 2;
    for (ssize_t i = start_i; i < end_i; ++i)
    {
        const double *action_ptr = actions_acc.data(i, 0, 0);
        envs[i].step(
            action_ptr,
            obs_ptr + obs_size * i * sizeof(double),
            rewards_ptr + i * sizeof(double),
            terminateds_ptr + i * sizeof(bool),
            truncateds_ptr + i * sizeof(bool));
    }
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<bool>, py::array_t<bool>> VectorizedCoopPush::step(py::array_t<double> actions)
{
    // Ensure the numpy array is in a C-style contiguous layout for safe pointer access.
    std::swap(m_current_obs, m_next_obs); // current obs has presumable been saved at this point
    auto actions_cstyle = py::array_t<double, py::array::c_style | py::array::forcecast>(actions);
    double *obs_ptr = m_next_obs.mutable_data();
    double *rewards_ptr = m_rewards.mutable_data();
    bool *terminated_ptr = m_terminateds.mutable_data();
    bool *truncated_ptr = m_truncateds.mutable_data();
    // Dispatch jobs to the thread pool, dividing the work into chunks.
    for (ssize_t i = 0; i < this->m_n_envs; i += this->m_envs_per_job)
    {
        const ssize_t start_i = i;
        const ssize_t end_i = std::min(i + (ssize_t)this->m_envs_per_job, (ssize_t)this->m_n_envs);

        this->m_pool.enqueue([this, &actions_cstyle, obs_ptr, rewards_ptr, terminated_ptr, truncated_ptr, start_i, end_i]
                             { step_job(
                                   actions_cstyle,
                                   obs_ptr,
                                   rewards_ptr,
                                   terminated_ptr,
                                   truncated_ptr,
                                   this->m_envs,
                                   start_i,
                                   end_i,
                                   this->m_env_obs_size); });
    }

    this->m_pool.wait_all();
    return std::make_tuple(m_next_obs, m_rewards, m_terminateds, m_truncateds);
}