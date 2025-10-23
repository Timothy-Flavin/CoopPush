import numpy as np
import time
from src.cooppush import environment as cooppush_cpp
from src.cooppush import vectorized_environment as cooppushvec
import torch
import torch.nn as nn
import pygame
import matplotlib.pyplot as plt
import random

N_ENV = 1152
N_THREADS = 6

d_to_c_map = [
    [0.0, 0.0],
    [0.0, -1.0],
    [1.0, -1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 0.0],
    [-1.0, 1.0],
]

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
    truncate_after=150,
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
    envs_per_job=8,
    truncate_after=150,
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
    fps=30,
)


# Try out 10,000 steps in each environment
def pure_environment_speed_test(num_steps=100000):
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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(23, 256)
        self.h2 = nn.Linear(256, 256)
        self.act_head = nn.Linear(256, 4 * 9)
        self.relu = nn.ReLU()
        self.double()

    def forward(self, x):
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.relu(self.act_head(x))
        v = []
        for s in range(x.ndim - 1):
            v.append(x.shape[s])
        v.append(4)
        v.append(9)
        return x.view(v)


def update_model(
    obs, next_obs, reward, terminated, actions, model, batch_size, max_idx, model_opt
):
    idx = torch.randint(0, high=max_idx, size=(batch_size,))
    # print(
    #     f"idx shape: {idx.shape}, actions shape: {actions.shape} obs shape: {obs.shape}"
    # )
    # for p in model.parameters():
    #    print(p)
    # print(f"model type: {model}, obs type: {obs[0:5]}")
    # input(
    #     f"obs shape: {obs[idx].shape} model(obs) shape: {model(obs[idx]).shape} action shape: {actions[idx].unsqueeze(-1).shape}"
    # )
    q_now = model(obs[idx]).gather(index=actions[idx].unsqueeze(-1), dim=-1).squeeze(-1)
    # print(f"Starting model update with q now shape {q_now.shape}")

    with torch.no_grad():
        q_next = model(next_obs[idx])
        # print(f"q_next shape: {q_next.shape}")
        # print(f"qmax shape: {q_next.max(dim=-1).values.shape}")
        # print(f"terminated shape: {terminated[idx].unsqueeze(-1).shape}")
        # print(f"reward shape: {reward[idx].unsqueeze(-1).shape}")
        target = (
            reward[idx].unsqueeze(-1)
            + 0.99 * terminated[idx].unsqueeze(-1) * q_next.max(dim=-1).values
        )
        # print(f"Target shape: {target.shape}")
    loss = ((q_now - target) ** 2).mean()
    model_opt.zero_grad()
    loss.backward()
    model_opt.step()
    return loss.item()


def _old_env_gpu(env, d_to_c_map, num_steps=10000):
    model = MLP().to("cuda")
    model_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    current_idx = 0
    obs_buffer = torch.zeros((10000, 23), device="cuda", dtype=torch.double)
    next_obs_buffer = torch.zeros((10000, 23), device="cuda", dtype=torch.double)
    reward_buffer = torch.zeros((10000), device="cuda", dtype=torch.double)
    terminated_buffer = torch.zeros((10000), device="cuda", dtype=torch.double)
    actions_buffer = torch.zeros((10000, 4), device="cuda", dtype=torch.long)

    print("Starting pure environment speed test...")
    # Single C++ environment
    obs, info = env.reset()
    r_hist = [0.0]
    l_hist = []
    # print(obs)

    start_time = time.time()
    for _ in range(num_steps):
        torch_obs = torch.from_numpy(obs["particle_0"]).double().to("cuda")
        # print(f"torch obs shape: {torch_obs}")
        with torch.no_grad():
            if random.random() > _ / num_steps:
                raw_actions = torch.randint(low=0, high=9, size=(4,))
            else:
                raw_actions = model(torch_obs).argmax(dim=-1).squeeze(0).detach()
        # print(f"raw action shape: {raw_actions.shape}")
        env_actions = {}
        for r in range(len(raw_actions)):
            env_actions[f"particle_{r}"] = d_to_c_map[raw_actions[r].to("cpu").item()]

        next_obs, reward, terminated, truncated, info = env.step(env_actions)
        # env.render()
        # pygame.event.clear()
        r_hist[-1] = r_hist[-1] + reward["particle_0"]
        obs_buffer[current_idx] = torch.from_numpy(obs["particle_0"]).to("cuda")
        next_obs_buffer[current_idx] = torch.from_numpy(next_obs["particle_0"]).to(
            "cuda"
        )
        reward_buffer[current_idx] = reward["particle_0"]
        terminated_buffer[current_idx] = float(terminated["particle_0"])
        actions_buffer[current_idx] = raw_actions

        if current_idx > 128:
            l_hist.append(
                update_model(
                    obs_buffer,
                    next_obs_buffer,
                    reward_buffer,
                    terminated_buffer,
                    actions_buffer,
                    model,
                    batch_size=128,
                    max_idx=min(_, 10000),
                    model_opt=model_opt,
                )
            )
        obs = next_obs
        current_idx += 1
        if current_idx == 10000:
            current_idx = 0

        if terminated["particle_0"] or truncated["particle_0"]:
            print(
                f"ep reward: {r_hist[-1]}, steps / sec: {_ / (time.time()-start_time):0.3f}"
            )
            r_hist.append(0.0)
            print("terminated or truncated so resetting")
            obs, info = env.reset()
    end_time = time.time()
    single_cpp_duration = end_time - start_time
    print(
        f"Single C++ environment: {num_steps} steps in {single_cpp_duration:.2f} seconds ({num_steps/single_cpp_duration:.2f} steps/sec)"
    )
    plt.plot(r_hist)
    plt.title("reward history traditional env")
    plt.show()
    plt.plot(l_hist)
    plt.title("loss hist traditional env")
    plt.show()


def _new_env_gpu(env, d_to_c_map, n_envs=N_ENV, n_threads=N_THREADS, num_steps=10000):
    model = MLP().to("cuda")
    # Enable kernel selection heuristics
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    model_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    current_idx = 0
    obs_buffer = torch.zeros((10000, n_envs, 23), device="cuda", dtype=torch.double)
    next_obs_buffer = torch.zeros(
        (10000, n_envs, 23), device="cuda", dtype=torch.double
    )
    reward_buffer = torch.zeros((10000, n_envs), device="cuda", dtype=torch.double)
    terminated_buffer = torch.zeros((10000, n_envs), device="cuda", dtype=torch.double)
    actions_buffer = torch.zeros((10000, n_envs, 4), device="cuda", dtype=torch.long)

    print("Starting pure environment speed test...")
    # Single C++ environment
    obs: np.ndarray = env.reset()
    print(f"obs shape: {obs.shape}")
    r_hist = []
    for i in range(n_envs):
        r_hist.append([0])
    l_hist = []
    # print(obs)

    start_time = time.time()
    # Preallocate pinned CPU buffer for fast, non-blocking H2D transfers
    pinned_cpu_obs = torch.empty((n_envs, 23), dtype=torch.float64, pin_memory=True)
    # Initial transfer
    pinned_cpu_obs.copy_(torch.from_numpy(obs).double())
    torch_obs = pinned_cpu_obs.unsqueeze(0).to("cuda", non_blocking=True)
    # Training hyperparameters to drive more GPU work per env step
    train_min_warmup = 64  # timesteps before starting updates
    train_steps_per_iter = 2  # number of optimizer steps per environment step
    train_batch_size = (
        16  # temporal samples; effective batch is train_batch_size * n_envs
    )
    for step_i in range(num_steps):
        # print(f"torch obs shape: {torch_obs.shape}")
        with torch.no_grad():
            if random.random() > step_i / num_steps:
                raw_actions = torch.randint(
                    low=0,
                    high=9,
                    size=(
                        n_envs,
                        4,
                    ),
                )
            else:
                raw_actions = model(torch_obs).argmax(dim=-1).detach()
                if raw_actions.ndim > 2:
                    raw_actions = raw_actions.squeeze(0)
        # input(
        #     f"raw action shape: {raw_actions.shape} buff shape: {actions_buffer[current_idx].shape}"
        # )

        env_actions = np.zeros((n_envs, 4, 2))
        # print(f"raw act shape: {raw_actions.shape}")
        for e in range(n_envs):
            # print(f"raw act: {raw_actions}")
            for a in range(4):
                # print(f"raw actions ea: {raw_actions[e, a]}")
                env_actions[e, a] = d_to_c_map[raw_actions[e, a]]
        next_obs, reward, terminated, truncated = env.step(env_actions)
        # Non-blocking H2D transfer via preallocated pinned buffer
        # pinned_cpu_obs.copy_(torch.from_numpy(next_obs).double(), non_blocking=True)
        # next_torch_obs = pinned_cpu_obs.unsqueeze(0).to("cuda", non_blocking=True)
        next_torch_obs = torch.from_numpy(next_obs).to("cuda")
        # env.render()
        # pygame.event.clear()
        for e in range(n_envs):
            r_hist[e][-1] += reward[e]
        obs_buffer[current_idx] = torch_obs
        next_obs_buffer[current_idx] = next_torch_obs
        reward_buffer[current_idx] = torch.from_numpy(reward).to("cuda")
        terminated_buffer[current_idx] = torch.from_numpy(terminated).to("cuda")
        actions_buffer[current_idx] = raw_actions

        if current_idx > train_min_warmup:
            # Perform multiple optimizer steps to increase GPU utilization
            for _t in range(train_steps_per_iter):
                l_hist.append(
                    update_model(
                        obs_buffer,
                        next_obs_buffer,
                        reward_buffer,
                        terminated_buffer,
                        actions_buffer,
                        model,
                        batch_size=train_batch_size,
                        max_idx=min(current_idx, 10000),
                        model_opt=model_opt,
                    )
                )
            # print(l_hist[-1])

        current_idx += 1
        if current_idx == 10000:
            current_idx = 0

        if terminated[0] or truncated[0]:
            print(
                f"ep reward env{0}: {r_hist[0][-1]}, steps / sec: {step_i*n_envs / (time.time()-start_time):0.3f}"
            )

        for e in range(n_envs):
            if terminated[e] or truncated[e]:
                r_hist[e].append(0.0)
                # Reset a single env with non-blocking transfer
                env.reset_i(e)
                # Copy into the appropriate slice of pinned buffer and then to GPU
                # pinned_cpu_obs[e].copy_(torch.from_numpy(reset_obs).double())
                # torch_obs[:, e] = pinned_cpu_obs[e].to("cuda", non_blocking=True)

        torch_obs = next_torch_obs
        obs = next_obs
    end_time = time.time()
    single_cpp_duration = end_time - start_time
    print(
        f"Single C++ environment: {num_steps} steps in {single_cpp_duration:.2f} seconds ({num_steps/single_cpp_duration:.2f} steps/sec)"
    )
    # plt.plot(r_hist)
    # plt.title("reward history traditional env")
    # plt.show()
    # plt.plot(l_hist)
    # plt.title("loss hist traditional env")
    # plt.show()


def training_and_env_speed(num_steps=10000):
    global N_ENV, N_THREADS, d_to_c_map

    # _old_env_gpu(env=cpp_single_env, d_to_c_map=d_to_c_map, num_steps=10000)

    # _new_env_gpu(cpp_vec1_env, d_to_c_map, n_envs=1, n_threads=1, num_steps=10000)
    _new_env_gpu(
        cpp_vec_env, d_to_c_map, n_envs=N_ENV, n_threads=N_THREADS, num_steps=10000
    )
    exit()
    # Vectorized C++ environment with 1 env
    print("Starting vec1 environment speed test...")
    cpp_vec1_env.reset()
    print("vec1 environment reset done.")
    start_time = time.time()
    for _ in range(num_steps):
        actions = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
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


if __name__ == "__main__":
    # pure_environment_speed_test(200000)
    training_and_env_speed()
