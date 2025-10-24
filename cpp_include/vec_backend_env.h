#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for std::vector conversion
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
namespace py = pybind11;
struct StepResult
{
    std::vector<float> observation;
    float reward;
    bool terminated;
    bool truncated;
};
class VecBackendEnv
{
private:
    // --- Member Variables ---
    int num_particles_ = 0;

    // Initial state (used for reset)
    std::vector<float> initial_particle_positions_;
    std::vector<float> initial_boulder_positions_;
    std::vector<float> initial_landmark_positions_;
    // Current state
    std::vector<float> current_particle_positions_;
    std::vector<float> current_boulder_positions_;
    std::vector<float> current_landmark_positions_;
    std::vector<float> next_particle_positions_;
    std::vector<float> next_boulder_positions_;
    std::vector<float> next_landmark_positions_;
    std::vector<float> current_particle_velocities_;
    std::vector<float> current_boulder_velocities_;

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
    float displacement_reward = 0.0;
    float sparse_weight = 1.0;
    float delta_time = 0.1;
    float boulder_weight = 5.0;
    int truncate_after_steps_ = 1000;
    int current_step = 0;

    // Constants used by the environment. Make these static constexpr so they
    // don't participate in the implicit copy/assignment operators of the
    // class (which previously deleted the assignment operator and caused
    // std::vector<VecBackendEnv> usages to fail).
    static constexpr float LANDMARK_R = 1.0;
    static constexpr float BOULDER_R = 5.0;
    static constexpr float PARTICLE_R = 1.0;
    static constexpr float total_radius = BOULDER_R + PARTICLE_R;
    static constexpr float total_radius_sq = total_radius * total_radius;

    static constexpr float TOTAL_BOULDER_R = 2.0 * BOULDER_R;
    static constexpr float TOTAL_BOULDER_R_SQ = TOTAL_BOULDER_R * TOTAL_BOULDER_R;
    int global_state_size;
    const int my_index;

public:
    VecBackendEnv();
    VecBackendEnv(std::vector<float> particle_positions,
                  std::vector<float> boulder_positions,
                  std::vector<float> landmark_positions,
                  int n_physics_steps,
                  bool sparse_rewards,
                  bool visit_all,
                  float sparse_weight,
                  float dt,
                  float boulder_weight,
                  int truncate_after_steps,
                  const int idx);
    ~VecBackendEnv() {};
    void reset(float *global_state_ptr);
    void get_global_state(float *global_state_ptr);
    void set_naive_next_pos(const float *actions);
    void move_things();
    float get_reward_all();
    float get_reward_one();
    int state_size();
    int get_num_particles() { return this->n_particles; };
    void step(const float *actions, float *obs_ptr, float *rewards_ptr, bool *terminateds_ptr, bool *truncateds_ptr);
};