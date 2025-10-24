import numpy as np
import time
from src.cooppush import environment as cooppush_cpp
from src.cooppush import vectorized_environment as cooppushvec
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import matplotlib.pyplot as plt
import random
from copy import deepcopy

N_ENV = 768
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
d_to_c_map = np.asarray(d_to_c_map)

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
    def update_model(
        obs,
        next_obs,
        reward,
        terminated,
        truncated,
        actions,
        model,
        target_model,
        batch_size,
        max_idx,
        model_opt,
        gamma: float = 0.99,
        grad_clip: float = 10.0,
    ):
        """One Double-DQN VDN update on a random time batch.

        Shapes:
          - obs,next_obs: [T, N, 23]
          - actions:      [T, N, 4] (each in [0..8])
          - reward, done: [T, N]
          Returns scalar loss.
        """
        idx = torch.randint(0, high=max_idx, size=(batch_size,), device=obs.device)

        # Q(s,a) for chosen actions, sum over agents (VDN)
        q_logits_now = model(obs[idx])               # [B, N, 4, 9]
        q_taken = q_logits_now.gather(
            dim=-1, index=actions[idx].unsqueeze(-1)
        ).squeeze(-1)                                # [B, N, 4]
        q_now = q_taken.sum(-1)                      # [B, N]

        with torch.no_grad():
            # Double DQN: select with online, evaluate with target
            q_logits_next_online = model(next_obs[idx])         # [B, N, 4, 9]
            a_star = q_logits_next_online.argmax(dim=-1)        # [B, N, 4]

            q_logits_next_target = target_model(next_obs[idx])  # [B, N, 4, 9]
            q_next_taken = q_logits_next_target.gather(
                dim=-1, index=a_star.unsqueeze(-1)
            ).squeeze(-1)                                       # [B, N, 4]
            q_next = q_next_taken.sum(-1)                        # [B, N]

            done = (terminated[idx] > 0.5) | (truncated[idx] > 0.5)
            discount = (~done).float()
            target = reward[idx] + gamma * discount * q_next     # [B, N]

        loss = F.smooth_l1_loss(q_now, target)
        model_opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        model_opt.step()
        return loss.item()
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
        self.relu = nn.Tanh()
        self.float()

    def forward(self, x):
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.act_head(x)
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
    # print(f"idx: {idx}")
    # print(reward[idx])

    q_now = (
        model(obs[idx])
        .gather(index=actions[idx].unsqueeze(-1), dim=-1)
        .squeeze(-1)
        .sum(-1)
    )
    # print(f"qnow shape: {q_now.shape}")
    print(f"actions: {actions[idx]}")
    with torch.no_grad():
        print(model(obs[idx]))
        print(model(next_obs[idx]))
        print("Q shape stuff")
        qn = model(next_obs[idx])
        print(f"raw shape: {qn.shape}")
        qn = qn.max(dim=-1).values
        print(f"max shape: {qn.shape}")
        qn = qn.sum(-1)
        print(f"vdn shape: {qn.shape}")
        q_next = model(next_obs[idx]).max(dim=-1).values.sum(-1)
        # print(f"qnext shape: {q_now.shape}")
        # print(f"reward shape: {reward[idx].shape}")
        # print(f"terminated shape: {terminated[idx].shape}")
        # print(f"qnext shape: {q_next.shape}")
        print("reward")
        print(reward[idx])
        target = reward[idx] + 0.9 * (1 - terminated[idx]) * q_next
        print(target)
        # print(f"target shape: {target.shape}")
    loss = ((q_now - target) ** 2).mean()
    model_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model_opt.step()

    # for j in range(10):
    #     q_now = (
    #         model(obs[idx])
    #         .gather(index=actions[idx].unsqueeze(-1), dim=-1)
    #         .squeeze(-1)
    #         .sum(-1)
    #     )
    #     loss = ((q_now - target) ** 2).mean()
    #     model_opt.zero_grad()
    #     loss.backward()
    #     model_opt.step()
    #     print(f"loss in weird situation {j}: {loss.item()}")
    #     print(f"qnow: {q_now.mean()} qnext {q_next.mean()}")

    input("update working?")
    return loss.item()


def _old_env_gpu(env, d_to_c_map, num_steps=10000):
    model = MLP().to("cuda")
    model_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    current_idx = 0
    obs_buffer = torch.zeros((10000, 23), device="cuda", dtype=torch.float32)
    next_obs_buffer = torch.zeros((10000, 23), device="cuda", dtype=torch.float32)
    reward_buffer = torch.zeros((10000), device="cuda", dtype=torch.float32)
    terminated_buffer = torch.zeros((10000), device="cuda", dtype=torch.float32)
    actions_buffer = torch.zeros((10000, 4), device="cuda", dtype=torch.long)

    print("Starting pure environment speed test...")
    # Single C++ environment
    obs, info = env.reset()
    r_hist = [0.0]
    l_hist = []
    # print(obs)

    start_time = time.time()
    for _ in range(num_steps):
        torch_obs = torch.from_numpy(obs["particle_0"]).float().to("cuda")
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


def _new_env_gpu(
    env: cooppushvec.CoopPushVectorizedEnv,
    d_to_c_map,
    n_envs=N_ENV,
    n_threads=N_THREADS,
    num_steps=10000,
):
    model = MLP().to("cuda")
    target_model = deepcopy(model).to("cuda")
    tau = 0.005  # soft update rate
    times = {
        "action": 0.0,
        "to_env": 0.0,
        "step": 0.0,
        "save": 0.0,
        "train": 0.0,
        "reset": 0.0,
    }
    # Enable kernel selection heuristics
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    model_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    current_idx = 0
    obs_buffer = torch.zeros((10000, n_envs, 23), device="cuda", dtype=torch.float32)
    next_obs_buffer = torch.zeros((10000, n_envs, 23), device="cuda", dtype=torch.float32)
    reward_buffer = torch.zeros((10000, n_envs), device="cuda", dtype=torch.float32)
    terminated_buffer = torch.zeros((10000, n_envs), device="cuda", dtype=torch.float32)
    truncated_buffer = torch.zeros((10000, n_envs), device="cuda", dtype=torch.float32)
    actions_buffer = torch.zeros((10000, n_envs, 4), device="cuda", dtype=torch.long)

    print("Starting pure environment speed test...")
    # Single C++ environment
    obs: np.ndarray = env.reset()
    print(f"obs shape: {obs.shape}")
    # input(f"obs buffer shape: {obs_buffer.shape}")
    r_hist = []
    for i in range(n_envs):
        r_hist.append([0])
    l_hist = []
    # print(obs)

    start_time = time.time()
    # Preallocate pinned CPU buffer for fast, non-blocking H2D transfers
    pinned_cpu_obs = torch.empty((n_envs, 23), dtype=torch.float32, pin_memory=True)
    # Initial transfer
    pinned_cpu_obs.copy_(torch.from_numpy(obs).astype(np.float32))
    torch_obs = pinned_cpu_obs.to("cuda", non_blocking=True).view((1, n_envs, -1))
    # Training hyperparameters to drive more GPU work per env step
    train_min_warmup = 64  # timesteps before starting updates
    train_steps_per_iter = 32  # number of optimizer steps per environment step
    train_batch_size = (
        2  # temporal samples; effective batch is train_batch_size * n_envs
    )
    num_updates = 0
    for step_i in range(num_steps):
        # print(f"torch obs shape: {torch_obs.shape}")
        a_t = time.time()
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
        times["action"] += time.time() - a_t
        # input(
        #     f"raw action shape: {raw_actions.shape} buff shape: {actions_buffer[current_idx].shape}"
        # )

        a_to = time.time()
        raw_actions_cpu = raw_actions.detach().cpu().numpy()
        env_actions = d_to_c_map[raw_actions_cpu]  # np.empty((n_envs, 4, 2))
        # print(raw_actions_cpu[0:5])
        # print(env_actions[0:5])
        # input("hurah?")
        # # print(f"raw act shape: {raw_actions.shape}")
        # for e in range(n_envs):
        #     # print(f"raw act: {raw_actions}")
        #     for a in range(4):
        #         # print(f"raw actions ea: {raw_actions[e, a]}")
        #         env_actions[e, a] = d_to_c_map[raw_actions[e, a]]
        times["to_env"] += time.time() - a_to

        a_env = time.time()
    next_obs, reward, terminated, truncated = env.step(env_actions)
        env.render(env_num=0)
        pygame.event.clear()
        times["step"] += time.time() - a_env
        # Non-blocking H2D transfer via preallocated pinned buffer
        # pinned_cpu_obs.copy_(torch.from_numpy(next_obs).double(), non_blocking=True)
        # next_torch_obs = pinned_cpu_obs.unsqueeze(0).to("cuda", non_blocking=True)

        t_buff = time.time()
    next_torch_obs = torch.from_numpy(next_obs).to("cuda").float().view((1, n_envs, -1))
        for e in range(n_envs):
            r_hist[e][-1] += reward[e]
    obs_buffer[current_idx] = torch_obs[0]
    next_obs_buffer[current_idx] = next_torch_obs[0]
        reward_buffer[current_idx] = torch.from_numpy(reward).to("cuda")
        terminated_buffer[current_idx] = torch.from_numpy(terminated).to("cuda")
    truncated_buffer[current_idx] = torch.from_numpy(truncated).to("cuda")
    actions_buffer[current_idx] = raw_actions

        times["save"] += time.time() - t_buff

        t_train = time.time()
        if current_idx > train_min_warmup:
            # Perform multiple optimizer steps to increase GPU utilization
            for _t in range(train_steps_per_iter):
                l_hist.append(
                    update_model(
                        obs_buffer,
                        next_obs_buffer,
                        reward_buffer,
                        terminated_buffer,
                        truncated_buffer,
                        actions_buffer,
                        model,
                        target_model,
                        batch_size=train_batch_size,
                        max_idx=min(current_idx, 10000),
                        model_opt=model_opt,
                        gamma=0.99,
                        grad_clip=10.0,
                    )
                )
                # Soft update target network
                with torch.no_grad():
                    for p_t, p in zip(target_model.parameters(), model.parameters()):
                        p_t.data.lerp_(p.data, tau)
                num_updates += 1
        times["train"] += time.time() - t_train
        # print(l_hist[-1])
        cached_idx = current_idx
        current_idx += 1
        if current_idx == 10000:
            current_idx = 0

        done = False
        if terminated[0] or truncated[0]:
            done = True

        t_reset = time.time()
        for e in range(n_envs):
            if terminated[e] or truncated[e]:
                r_hist[e].append(0.0)
                # Reset a single env with non-blocking transfer
                env.reset_i(e)
                next_torch_obs = (
                    torch.from_numpy(next_obs).to("cuda").view((1, n_envs, -1))
                )
                # Copy into the appropriate slice of pinned buffer and then to GPU
                # pinned_cpu_obs[e].copy_(torch.from_numpy(reset_obs).double())
                # torch_obs[:, e] = pinned_cpu_obs[e].to("cuda", non_blocking=True)

        times["reset"] += time.time() - t_reset

        if done:
            print(
                f"ep reward env{0}: {r_hist[0][-1]}, steps / sec: {step_i*n_envs / (time.time()-start_time):0.3f}"
            )
            print(f"current epsilon: {step_i / num_steps}")
            print(times)

        #     print(f"torch obs shape: {torch_obs.shape}")
        #     print(f"buffer idx: {cached_idx} env elements: ")
        #     print(f"  actions: {raw_actions_cpu[0]}")
        #     print(f"  obs: {torch_obs[0,0]}")
        #     print(f"  numpy obs: {obs[0]}")
        #     print(f"  next_obs: {next_torch_obs[0,0]}")
        #     print(f"  numpy next_obs: {next_obs[0]}")
        #     print(f"  reward: {reward[0]}")
        #     print(f"  term: {terminated[0]}")
        #     print(f"  trunc: {truncated[0]}")
        #     print("-----------BUFFER ITEMS-------")
        #     print(f"  buff_obs: {obs_buffer[cached_idx,0]}")
        #     print(f"  buff_next_obs: {next_obs_buffer[cached_idx,0]}")
        #     print(f"  buff_reward: {reward_buffer[cached_idx,0]}")
        #     print(f"  buff_terminated: {terminated_buffer[cached_idx,0]}")
        #     print(f"  buff_truncated: {truncated_buffer[cached_idx,0]}")
        #     print()
        #     input("Continue? ")

        torch_obs = next_torch_obs
        obs = next_obs
    end_time = time.time()
    single_cpp_duration = end_time - start_time
    print(
        f"Single C++ environment: {num_steps} steps in {single_cpp_duration:.2f} seconds ({num_steps/single_cpp_duration:.2f} steps/sec)"
    )

    max_len = len(r_hist[0])
    for e in range(n_envs):
        if len(r_hist[e]) > max_len:
            max_len = r_hist[e]

    res_raw = np.empty((len(r_hist), max_len))
    for e in range(n_envs):
        res_raw[: len(r_hist[e])] = np.array(r_hist[e])

    mean_rs = res_raw.mean(axis=0)
    top = res_raw.max(axis=0)
    bot = res_raw.min(axis=0)

    plt.plot(top)
    plt.plot(mean_rs)
    plt.plot(bot)
    plt.legend(["max", "mean", "min"])
    plt.title("reward history traditional env")
    plt.show()
    plt.plot(l_hist)
    plt.title("loss hist traditional env")
    plt.show()


def training_and_env_speed(num_steps=10000):
    global N_ENV, N_THREADS, d_to_c_map

    # _old_env_gpu(env=cpp_single_env, d_to_c_map=d_to_c_map, num_steps=10000)

    # _new_env_gpu(cpp_vec1_env, d_to_c_map, n_envs=1, n_threads=1, num_steps=10000)
    _new_env_gpu(
        cpp_vec_env, d_to_c_map, n_envs=N_ENV, n_threads=N_THREADS, num_steps=2000
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
