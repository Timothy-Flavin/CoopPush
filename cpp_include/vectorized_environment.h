#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for std::vector conversion
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
#include "thread_pool.h"
namespace py = pybind11;
#include "vec_backend_env.h"

class VectorizedCoopPush
{
private:
    // --- Member Variables ---
    // The *actual* memory owners
    py::array_t<double> m_buffer_1;
    py::array_t<double> m_buffer_2;
    // "Pointers" to the current role of each buffer
    py::array_t<double> m_current_obs;
    py::array_t<double> m_next_obs;
    // Buffers for rewards, terminals, etc.
    py::array_t<double> m_rewards;
    py::array_t<bool> m_terminateds;
    py::array_t<bool> m_truncateds;

    ThreadPool m_pool;
    std::vector<VecBackendEnv> m_envs;
    int m_n_envs = 1;
    int m_envs_per_job = 1;
    int m_n_threads = 1;
    int m_env_obs_size = 10;

public:
    VectorizedCoopPush();
    VectorizedCoopPush(std::vector<double> particle_positions,
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
                       int envs_per_job);
    ~VectorizedCoopPush() {};
    py::array_t<double> reset();
    py::array_t<double> reset_i(int i);
    inline int obs_size() { return m_env_obs_size; };
    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<bool>, py::array_t<bool>> step(py::array_t<double> actions);
};