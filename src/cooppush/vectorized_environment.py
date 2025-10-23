import json
from typing import Tuple

import numpy as np

import cooppush.cooppush_cpp as cooppush_cpp


class CoopPushVectorizedEnv:
    """
    Vectorized wrapper around the C++ VectorizedEnvironment.

    - Manages N independent environments in C++.
    - Accepts actions as a numpy array with shape (n_envs, n_particles, 2).
    - Returns batched numpy arrays for state, reward, terminations, and truncations.

    Notes:
    - The C++ backend returns the global state per environment (flattened array).
    - Optionally normalizes returned states if normalize_observations=True.
    """

    def __init__(
        self,
        json_path: str = "default_push_level.json",
        num_envs: int = 128,
        num_threads: int = 2,
        envs_per_job: int = 64,
        truncate_after=200,
        cpp_steps_per_step: int = 10,
        sparse_rewards: bool = True,
        visit_all: bool = True,
        sparse_weight: float = 5.0,
        dt: float = 0.2,
        boulder_weight: float = 4.0,
        normalize_observations: bool = True,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_threads = int(num_threads)
        self.cpp_steps_per_step = int(cpp_steps_per_step)
        self.sparse_rewards = bool(sparse_rewards)
        self.visit_all = bool(visit_all)
        self.sparse_weight = float(sparse_weight)
        self.dt = float(dt)
        self.boulder_weight = float(boulder_weight)
        self.normalize_observations = bool(normalize_observations)

        with open(json_path) as f:
            env_setup = json.load(f)

        # Topology sizes
        self.n_particles = len(env_setup["particle_pos"]) // 2
        self.n_boulders = len(env_setup["boulder_pos"]) // 2
        self.n_landmarks = len(env_setup["landmark_pos"]) // 2

        # Keep the starts as float64 arrays (CPP expects double)
        self._initial_particle_pos = np.array(
            env_setup["particle_pos"], dtype=np.float64
        )
        self._initial_boulder_pos = np.array(env_setup["boulder_pos"], dtype=np.float64)
        self._initial_landmark_pos = np.array(
            env_setup["landmark_pos"], dtype=np.float64
        )

        # Initialize C++ vectorized env
        # Binding signature (per backend.cpp):
        self.cpp_env = cooppush_cpp.VectorizedEnvironment(
            particle_positions=self._initial_particle_pos,
            boulder_positions=self._initial_boulder_pos,
            landmark_positions=self._initial_landmark_pos,
            n_physics_steps=self.cpp_steps_per_step,
            sparse_rewards=self.sparse_rewards,
            visit_all=self.visit_all,
            sparse_weight=self.sparse_weight,
            dt=self.dt,
            boulder_weight=self.boulder_weight,
            truncate_after_steps=truncate_after,
            n_threads=num_threads,
            n_envs=num_envs,
            envs_per_job=envs_per_job,
        )

        # Precompute normalization vector if requested
        self._norm_array = None
        if self.normalize_observations:
            v_every_size = (
                self.n_boulders * self.n_landmarks
                if self.visit_all
                else self.n_boulders
            )
            state_dim = (
                self.n_particles * 4
                + self.n_boulders * 2
                + self.n_landmarks * 2
                + v_every_size
            )
            self.state_dim = state_dim
            norm_array = np.ones(state_dim, dtype=np.float32)
            for i in range(self.n_particles):
                norm_array[i * 4] = 25.0
                norm_array[i * 4 + 1] = 25.0
                norm_array[i * 4 + 2] = 1.0
                norm_array[i * 4 + 3] = 1.0
            for i in range(self.n_boulders):
                idx = self.n_particles * 4 + i * 2
                norm_array[idx] = 25.0
                norm_array[idx + 1] = 25.0
            for i in range(self.n_landmarks):
                idx = self.n_particles * 4 + self.n_boulders * 2 + i * 2
                norm_array[idx] = 25.0
                norm_array[idx + 1] = 25.0
            self._norm_array = norm_array

    def reset(self) -> np.ndarray:
        """
        Resets all vectorized environments.

        Returns
        -------
        states: np.ndarray
            Array of shape (num_envs, state_dim) with the global state per environment.
        """
        print("python resetting")
        obs = self.cpp_env.reset()
        # print("python wrapper got states from cpp")
        if self.normalize_observations and self._norm_array is not None:
            obs = obs / self._norm_array[np.newaxis, :]
        return obs

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Steps all environments with vectorized actions.

        Parameters
        ----------
        actions : np.ndarray
            Shape (num_envs, n_particles, 2), dtype float32/float64.

        Returns
        -------
        next_states : np.ndarray
            Shape (num_envs, state_dim), global state per environment.
        rewards : np.ndarray
            Shape (num_envs,), scalar reward per environment.
        terminations : np.ndarray
            Shape (num_envs,), boolean flags for termination per environment.
        truncations : np.ndarray
            Shape (num_envs,), boolean flags for truncation per environment.
        """
        assert (
            actions.ndim == 3
            and actions.shape[1] == self.n_particles
            and actions.shape[2] == 2
        ), f"actions must have shape (num_envs={self.num_envs}, n_particles={self.n_particles}, 2) but had shape {actions.shape}"

        # Ensure double precision for C++ side; no copy if already contiguous float64
        actions_np = np.asarray(actions, dtype=np.float64)
        # states_array, rewards, terminations, truncations = self.cpp_env.step(actions_np)
        obs, reward, term, trunc = self.cpp_env.step(actions_np)

        if self.normalize_observations and self._norm_array is not None:
            obs = obs / self._norm_array[np.newaxis, :]

        self.active_buffer = 1 - self.active_buffer
        return obs, reward, term, trunc

    def close(self) -> None:
        # Nothing to do: C++ objects release on GC; keep for API symmetry
        pass

    def reset_i(self, i):
        obs = self.cpp_env.reset_i(i)
        if self.normalize_observations and self._norm_array is not None:
            obs[i * self.state_dim : (i + 1) * self.state_dim] = (
                obs[i * self.state_dim : (i + 1) * self.state_dim]
                / self._norm_array[np.newaxis, i, :]
            )
        return states


if __name__ == "__main__":
    # Minimal smoke test
    import torch

    N_ENV = 4
    N_THREAD = 2
    env = CoopPushVectorizedEnv(
        num_envs=N_ENV, num_threads=N_THREAD, cpp_steps_per_step=5
    )
    S_DIM = env.state_dim
    print("resetting cpp env")
    states = env.reset()
    n_envs = env.num_envs
    actions = np.random.uniform(-1, 1, size=(n_envs, env.n_particles, 2)).astype(
        np.float64
    )

    # simulate mem buffer simulation
    state_buff = torch.zeros((10, N_ENV, S_DIM), dtype=torch.float64, device="cuda")
    next_state_buff = torch.zeros(
        (10, N_ENV, S_DIM), dtype=torch.float64, device="cuda"
    )
    reward_buff = torch.zeros((10, N_ENV), dtype=torch.float64, device="cuda")
    term_buff = torch.zeros((10, N_ENV), dtype=torch.bool, device="cuda")
    trunc_buff = torch.zeros((10, N_ENV), dtype=torch.bool, device="cuda")

    torch_states = torch.from_numpy(states).to("cuda")
    for i in range(5):
        next_states, rewards, terms, truncs = env.step(actions)
        torch_next_states = torch.from_numpy(next_states).to("cuda")

        state_buff[i] = torch_states
        next_state_buff[i] = torch_next_states
        reward_buff[i] = torch.from_numpy(rewards).to("cuda")
        term_buff[i] = torch.from_numpy(terms).to("cuda")

        for e in range(N_ENV):
            if terms[e] or truncs[e]:
                print(f"resetting {e}, will show up in obs but not buffer")
                env.reset_i(e)

        print("-------------------CURRENT-ENV-VARIABLES-----------------------")
        print(f"  Step: {i} complete")
        print(f"  states: shape {states.shape} ; {states}")
        print(f"  next_states: shape {next_states.shape} ; {next_states}")
        print(f"  rewards: shape {rewards.shape} ; {rewards}")
        print(f"  terms: shape {terms.shape} ; {terms}")
        print(f"  truncs: shape {truncs.shape} ; {truncs}")
        print()
        print("-------------------BUFFER-ENV-VARIABLES------------------------")
        print(f"  state buffer: {state_buff}")
        print(f"  next_states buffer: {next_state_buff}")
        print(f"  rewards buffer: {reward_buff}")
        print(f"  term buffer: {term_buff}")
        print(f"  trunc buffer: {trunc_buff}")
        print("\n\n")
