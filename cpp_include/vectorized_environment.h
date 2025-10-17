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
    std::vector<std::vector<double>> global_state_vecs;
    std::vector<double> rewards;
    std::vector<bool> terminateds;
    std::vector<bool> truncateds;
    ThreadPool pool;
    std::vector<VecBackendEnv> envs;
    int n_envs = 1;
    int envs_per_job = 1;
    int n_threads = 1;
    int env_obs_size = 10;

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
    ~VectorizedCoopPush();
    py::array_t<double> reset();
    py::tuple step(py::array_t<double> actions);
};