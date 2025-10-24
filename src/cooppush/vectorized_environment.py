import json
from typing import Tuple

import numpy as np
import pygame
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
        render_mode: str | None = None,
    ) -> None:
        self.render_mode = render_mode
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
        self.particle_radius = 1
        self.landmark_radius = 1
        self.boulder_radius = 5
        # self.randomize_order = randomize_order
        # self.start_noise = start_noise
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
            print(
                f"n boulder: {self.n_boulders} nlandmark: {self.n_landmarks}, vevery: {self.visit_all} size: {v_every_size}"
            )
            print(
                f"npart: {self.n_particles} nb: {self.n_boulders} nl: {self.n_landmarks}"
            )
            state_dim = (
                self.n_particles * 4
                + self.n_boulders * 2
                + self.n_landmarks * 2
                + v_every_size
            )
            self.state_dim = state_dim
            # Match C++ dtype (double) so in-place ops don't force copies
            norm_array = np.ones(state_dim, dtype=np.float64)
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

        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.clock = None
        if self.render_mode is not None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Particle Simulation")
            self.font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()
            self.fps = 30

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
        if self.render_mode is not None:
            self.cached_state = np.copy(obs)
        # If normalizing, do it in-place to preserve reference to C++ buffer
        if self.normalize_observations and self._norm_array is not None:
            np.divide(obs, self._norm_array[np.newaxis, :], out=obs, casting="unsafe")
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
        if self.render_mode is not None:
            self.cached_state = np.copy(obs)
        if self.normalize_observations and self._norm_array is not None:
            # In-place to keep sharing C++ buffer
            np.divide(obs, self._norm_array[np.newaxis, :], out=obs, casting="unsafe")
        return obs, reward, term, trunc

    def close(self) -> None:
        # Nothing to do: C++ objects release on GC; keep for API symmetry
        pass

    def reset_i(self, i):
        # Returns a view of the i-th environment's state (1D), still referencing C++ memory
        obs = self.cpp_env.reset_i(i)
        if self.normalize_observations and self._norm_array is not None:
            # Normalize only the i-th row in-place, preserving reference semantics
            np.divide(obs[i], self._norm_array, out=obs[i], casting="unsafe")
        return obs[i]

    def scale_to_screen(self, x, y):
        x = (x - self.min_x) / self.x_range * self.screen_width
        y = (y - self.min_y) / self.y_range * self.screen_height
        return (int(x), int(y))

    def scale_screen(self, _state):
        self.min_x = _state[0]
        self.max_x = _state[0]
        self.min_y = _state[1]
        self.max_y = _state[1]
        for i in range(_state.shape[0]):
            if i % 2 == 0:
                if _state[i] < self.min_x:
                    self.min_x = _state[i]
                if _state[i] > self.max_x:
                    self.max_x = _state[i]
            else:
                if _state[i] < self.min_y:
                    self.min_y = _state[i]
                if _state[i] > self.max_y:
                    self.max_y = _state[i]
        self.min_x = self.min_x - 10.0
        self.min_y = self.min_y - 10.0
        self.max_x = self.max_x + 10.0
        self.max_y = self.max_y + 10.0
        self.x_range = self.max_x - self.min_x
        self.y_range = self.max_y - self.min_y

        if self.x_range / self.screen_width > self.y_range / self.screen_height:
            avg_y = (self.min_y + self.max_y) / 2
            self.scale = self.x_range / self.screen_width
            self.min_y = avg_y - self.scale * self.screen_height / 2
            self.max_y = avg_y + self.scale * self.screen_height / 2
            self.y_range = self.max_y - self.min_y
        else:
            avg_x = (self.min_x + self.max_x) / 2
            self.scale = self.y_range / self.screen_height
            self.min_x = avg_x - self.scale * self.screen_width / 2
            self.max_x = avg_x + self.scale * self.screen_width / 2
            self.x_range = self.max_x - self.min_x

    def render(self, importance: None | np.ndarray = None, env_num=0) -> None | str:
        """
        Renders the environment to the screen using Pygame.

        This function assumes self.state is a flattened array of positions:
        [p1_x, p1_y, p2_x, p2_y, ..., b1_x, b1_y, ..., l1_x, l1_y, ...]
        """
        if self.render_mode != "human":
            # Do nothing if not in human rendering mode
            return
        assert self.screen is not None, "cant render to no screen"
        assert self.clock is not None, "cant tick nonexistent clock"
        # Colors for different objects (in RGB format)
        PARTICLE_COLOR = (255, 0, 0)  # Red
        BOULDER_COLOR = (128, 128, 128)  # Gray
        LANDMARK_COLOR = (0, 0, 255)  # Blue
        BACKGROUND_COLOR = (0, 0, 0)  # Black
        cs = self.cached_state[env_num]
        self.scale_screen(cs)
        # Clear the screen with the background color
        self.screen.fill(BACKGROUND_COLOR)

        # Draw particles
        for i in range(self.n_particles):
            # Calculate the index for the particle's x and y coordinates
            my_col = PARTICLE_COLOR
            if importance is not None:
                my_col = tuple(int(c * importance[i]) for c in PARTICLE_COLOR)

            x_idx = i * 4
            y_idx = i * 4 + 1
            x = cs[x_idx]
            y = cs[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.particle_radius / self.scale)
            pygame.draw.circle(self.screen, my_col, center, radius)
            # Render agent id as text on agent
            agent_label = self.font.render(str(i), True, (255, 255, 255))
            label_rect = agent_label.get_rect(center=center)
            self.screen.blit(agent_label, label_rect)
        # Draw boulders
        for i in range(self.n_boulders):
            # Calculate the index for the boulder's x and y coordinates
            offset = self.n_particles * 4
            x_idx = offset + i * 2
            y_idx = offset + i * 2 + 1
            x = cs[x_idx]
            y = cs[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.boulder_radius / self.scale)
            pygame.draw.circle(self.screen, BOULDER_COLOR, center, radius)

        # Draw landmarks
        for i in range(self.n_landmarks):
            # Calculate the index for the landmark's x and y coordinates
            offset = self.n_particles * 4 + self.n_boulders * 2
            x_idx = offset + i * 2
            y_idx = offset + i * 2 + 1
            x = cs[x_idx]
            y = cs[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.landmark_radius / self.scale)
            pygame.draw.circle(self.screen, LANDMARK_COLOR, center, radius)

        # Update the display to show the changes
        pygame.display.flip()

        # Control the frame rate
        # self.clock.tick(self.fps)
        # pygame.event.clear()


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
        if i == 2:
            terms[2] = True
        if i == 3:
            truncs[3] = True
        torch_next_states = torch.from_numpy(next_states).to("cuda")

        state_buff[i] = torch_states
        next_state_buff[i] = torch_next_states
        reward_buff[i] = torch.from_numpy(rewards).to("cuda")
        term_buff[i] = torch.from_numpy(terms).to("cuda")
        trunc_buff[i] = torch.from_numpy(truncs).to("cuda")

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
        input("next step")
        torch_states = torch_next_states
        states = next_states
