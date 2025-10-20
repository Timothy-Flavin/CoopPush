#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for std::vector conversion
#include <pybind11/numpy.h>
#include "vectorized_environment.h"
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
                                       int envs_per_job) : pool((ssize_t)n_threads)
{
    this->n_envs = n_envs;
    this->n_threads = n_threads;
    this->envs_per_job = envs_per_job;
    for (int i = 0; i < n_envs; ++i)
    {
        this->envs.push_back(VecBackendEnv(
            particle_positions,
            boulder_positions,
            landmark_positions,
            n_physics_steps,
            sparse_rewards,
            visit_all,
            sparse_weight,
            dt,
            boulder_weight,
            truncate_after_steps));
    }
    assert(!this->envs.empty());
    this->env_obs_size = this->envs.at(0).state_size();
    this->global_state_vecs.resize(n_envs, std::vector<double>(this->env_obs_size));
    this->rewards.resize(n_envs, 0.0);
    this->truncateds.resize(n_envs, false);
    this->terminateds.resize(n_envs, false);
}
py::array_t<double> VectorizedCoopPush::reset()
{
    std::cout << "C++ Vec ENV manager resetting" << std::endl;
    for (int i = 0; i < this->n_envs; ++i)
    {
        std::cout << "  resetting env: " << i << " of " << this->n_envs << std::endl;
        this->global_state_vecs[i] = this->envs[i].reset();
    }
    return vec2d_to_pyarray(this->global_state_vecs);
}

py::array_t<double> VectorizedCoopPush::reset_i(int i)
{
    this->global_state_vecs[i] = this->envs[i].reset();
    return vec1d_to_pyarray(this->global_state_vecs[i]);
}

void step_job(
    py::array_t<double, py::array::c_style | py::array::forcecast> &actions,
    std::vector<std::vector<double>> &observations,
    std::vector<double> &rewards,
    std::vector<bool> &terminateds,
    std::vector<bool> &truncateds,
    std::vector<VecBackendEnv> &envs,
    ssize_t start_i,
    ssize_t end_i)
{
    auto actions_acc = actions.unchecked<3>(); // Shape: [n_env, n_agent, act_size]
    ssize_t num_agents = envs[0].get_num_particles();
    ssize_t action_size = 2;
    for (ssize_t i = start_i; i < end_i; ++i)
    {
        const double *action_ptr = actions_acc.data(i, 0, 0);
        // Call the step function on the individual environment backend
        StepResult result = envs[i].step(action_ptr);

        // Store the results in the correct slice of the global vectors
        observations[i] = result.observation;
        rewards[i] = result.reward;
        terminateds[i] = result.terminated;
        truncateds[i] = result.truncated;
    }
}

py::tuple VectorizedCoopPush::step(py::array_t<double> actions)
{
    // Ensure the numpy array is in a C-style contiguous layout for safe pointer access.
    auto actions_cstyle = py::array_t<double, py::array::c_style | py::array::forcecast>(actions);

    // Dispatch jobs to the thread pool, dividing the work into chunks.
    for (ssize_t i = 0; i < this->n_envs; i += this->envs_per_job)
    {
        const ssize_t start_i = i;
        const ssize_t end_i = std::min(i + (ssize_t)this->envs_per_job, (ssize_t)this->n_envs);

        this->pool.enqueue([this, &actions_cstyle, start_i, end_i]
                           { step_job(
                                 actions_cstyle,
                                 this->global_state_vecs,
                                 this->rewards,
                                 this->terminateds,
                                 this->truncateds,
                                 this->envs,
                                 start_i,
                                 end_i); });
    }

    this->pool.wait_all();
    return py::make_tuple(
        py::cast(this->global_state_vecs),
        py::cast(this->rewards),
        py::cast(this->terminateds),
        py::cast(this->truncateds));
}