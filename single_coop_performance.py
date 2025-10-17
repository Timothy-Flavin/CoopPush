import numpy as np
import time
from src.cooppush import environment as cooppush_cpp
from src.cooppush import vectorized_environment as cooppushvec

N_ENV = 4
N_THREADS = 4

env_attributes = {
    "particle_pos": [5.0, 5.0, -5.0, 5.0, -5.0, -5.0, 5.0, -5.0],
    "boulder_pos": [0.0, 0.0],
    "landmark_pos": [20.0, 0.0, -20.0, 0.0],
}

cpp_vec1_env = cooppushvec.CoopPushVectorizedEnv(
    json_path="default_push_level.json",
    num_envs=1,
    num_threads=1,
    cpp_steps_per_step=10,
    sparse_rewards=False,
    visit_all=False,
    sparse_weight=1,
    dt=0.2,
    boulder_weight=1.0,
    normalize_observations=False,
)

cpp_vec_env = cooppushvec.CoopPushVectorizedEnv(
    json_path="default_push_level.json",
    num_envs=N_ENV,
    num_threads=N_THREADS,
    cpp_steps_per_step=10,
    sparse_rewards=False,
    visit_all=False,
    sparse_weight=1,
    dt=0.2,
    boulder_weight=1.0,
    normalize_observations=False,
)

cpp_single_env = cooppush_cpp.CoopPushEnv(
    json_path="default_push_level.json",
    cpp_steps_per_step=10,
    sparse_rewards=False,
    visit_all=False,
    sparse_weight=1,
    dt=0.2,
    boulder_weight=1.0,
    normalize_observations=False,
)


# Try out 10,000 steps in each environment
def pure_environment_speed_test(num_steps=10000):
    global N_ENV, N_THREADS
    print("Starting pure environment speed test...")
    # Single C++ environment
    cpp_single_env.reset()
    start_time = time.time()
    for _ in range(num_steps):
        actions = {
            "particle_0": [1.0, 0.0],
            "particle_1": [1.0, 0.0],
            "particle_2": [1.0, 0.0],
            "particle_3": [1.0, 0.0],
        }
        obs, reward, terminated, truncated, info = cpp_single_env.step(actions)
        if terminated or truncated:
            obs, info = cpp_single_env.reset()
    end_time = time.time()
    single_cpp_duration = end_time - start_time
    print(
        f"Single C++ environment: {num_steps} steps in {single_cpp_duration:.2f} seconds ({num_steps/single_cpp_duration:.2f} steps/sec)"
    )

    # Vectorized C++ environment with 1 env
    print("Starting vec1 environment speed test...")

    cpp_vec1_env.reset()
    print("vec1 environment reset done.")
    start_time = time.time()
    for _ in range(num_steps):
        actions = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        print(actions.shape)
        obs, reward, done, info = cpp_vec1_env.step(actions)
    end_time = time.time()
    vec1_cpp_duration = end_time - start_time
    print(
        f"Vectorized C++ environment (1 env): {num_steps} steps in {vec1_cpp_duration:.2f} seconds ({num_steps/vec1_cpp_duration:.2f} steps/sec)"
    )

    print(f"Using {N_ENV} environments and {N_THREADS} threads for vectorized env.")
    # Vectorized C++ environment with 4 envs
    cpp_vec_env.reset()
    start_time = time.time()
    for _ in range(num_steps // N_ENV):
        actions = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]] * N_ENV)
        obs, reward, done, info = cpp_vec_env.step(actions)
    end_time = time.time()
    vec_cpp_duration = end_time - start_time
    print(
        f"Vectorized C++ environment ({N_ENV} envs): {num_steps} steps in {vec_cpp_duration:.2f} seconds ({num_steps/vec_cpp_duration:.2f} steps/sec)"
    )


def training_and_env_speed():
    pass


if __name__ == "__main__":
    pure_environment_speed_test()
    training_and_env_speed()
