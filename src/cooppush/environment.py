import gymnasium
import numpy as np
import json
from gymnasium.spaces import Box
import pygame

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ActionType, AgentID, ObsType
import cooppush.cooppush_cpp as cooppush_cpp


# =============================================================================
# PETTINGZOO ENVIRONMENT WRAPPER
# =============================================================================
class CoopPushEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for the multi-particle push environment.

    This Python class handles the PettingZoo API, while the core logic is
    delegated to a C++ backend.
    """

    metadata = {
        "name": "multi_particle_push_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        json_path="default_push_level.json",
        render_mode: str | None = None,
        fps=10,
        sparse_rewards=True,
        visit_all=True,
        sparse_weight=5.0,
    ):
        super().__init__()
        self.continuous_actions = True
        self.fps = fps
        with open(json_path) as f:
            env_setup = json.load(f)

        env = cooppush_cpp.Environment()
        env.init(
            env_setup["particle_pos"],
            env_setup["boulder_pos"],
            env_setup["landmark_pos"],
            5,
            sparse_rewards,
            visit_all,
            sparse_weight=sparse_weight,
        )
        print(env)
        self.env = env
        self.n_particles = len(env_setup["particle_pos"]) // 2
        self.n_boulders = len(env_setup["boulder_pos"]) // 2
        self.n_landmarks = len(env_setup["landmark_pos"]) // 2
        self.particle_radius = 1
        self.landmark_radius = 1
        self.boulder_radius = 5

        self.render_mode = render_mode
        # Persistent variables for rendering
        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.clock = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Particle Simulation")
            self.clock = pygame.time.Clock()

        # --- PettingZoo API Requirements ---
        self.agents = [f"particle_{i}" for i in range(self.n_particles)]
        self.possible_agents = self.agents[:]

        # Define observation and action spaces for each agent
        # Each agent observes its own (x, y) position
        self.observation_spaces = {
            agent: Box(
                low=0,
                high=1,
                shape=(
                    self.n_particles * 4 + self.n_boulders * 2 + self.n_landmarks * 2,
                ),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        if self.continuous_actions:
            # Each agent has a 2D action: (dx, dy)
            self.action_spaces = {
                agent: Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            # 0: no-op, 1: right, 2: left, 3: up, 4: down
            self.action_spaces = {
                agent: gymnasium.spaces.Discrete(8) for agent in self.possible_agents
            }

        # --- State Caching for Rendering ---
        # This variable will hold the full state returned by the C++ backend
        # so the `render` function can use it without making another C++ call.
        self.cached_state = np.zeros(100)

    # Note: PettingZoo uses @functools.lru_cache(maxsize=None) for these properties
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict]:
        """Resets the environment and returns initial observations."""
        # The C++ backend handles the actual reset logic
        initial_state, initial_obs = self.env.reset()

        # --- Cache the state for rendering ---
        self.cached_state = initial_state

        # Reset the list of active agents
        self.agents = self.possible_agents[:]

        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return initial_obs, infos

    def step(self, actions: ActionType) -> tuple[
        ObsType,
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """
        Steps the environment.

        1. Formats actions for the backend.
        2. Calls the backend's step function.
        3. Caches the new state for rendering.
        4. Returns results in PettingZoo format.
        """

        # --- 2. Call the backend ---
        new_state, obs, rewards, terminations, truncations = self.env.step(actions)
        # if rewards["particle_0"] != 0:
        #    print(rewards)
        # --- 3. Cache the new state ---
        self.cached_state = np.copy(new_state)

        # --- 4. Format results for PettingZoo ---
        # Handle agent termination
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                pass

        # If all agents are done, clear the agents list for the next reset
        if not any(
            agent in self.agents
            for agent in self.possible_agents
            if not (terminations[agent] or truncations[agent])
        ):
            self.agents.clear()

        # Add the global state to the info dict for CTDE algorithms
        infos = {
            agent: {"global_state": self.cached_state} for agent in self.possible_agents
        }

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def scale_to_screen(self, x, y):
        x = (x - self.min_x) / self.x_range * self.screen_width
        y = (y - self.min_y) / self.y_range * self.screen_height
        return (int(x), int(y))

    def scale_screen(self):
        self.min_x = self.cached_state[0]
        self.max_x = self.cached_state[0]
        self.min_y = self.cached_state[1]
        self.max_y = self.cached_state[1]
        for i in range(self.cached_state.shape[0]):
            if i % 2 == 0:
                if self.cached_state[i] < self.min_x:
                    self.min_x = self.cached_state[i]
                if self.cached_state[i] > self.max_x:
                    self.max_x = self.cached_state[i]
            else:
                if self.cached_state[i] < self.min_y:
                    self.min_y = self.cached_state[i]
                if self.cached_state[i] > self.max_y:
                    self.max_y = self.cached_state[i]
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

    def render(self) -> None | str:
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

        self.scale_screen()
        # Clear the screen with the background color
        self.screen.fill(BACKGROUND_COLOR)

        # Draw particles
        for i in range(self.n_particles):
            # Calculate the index for the particle's x and y coordinates
            x_idx = i * 4
            y_idx = i * 4 + 1
            x = self.cached_state[x_idx]
            y = self.cached_state[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.particle_radius / self.scale)
            pygame.draw.circle(self.screen, PARTICLE_COLOR, center, radius)

        # Draw boulders
        for i in range(self.n_boulders):
            # Calculate the index for the boulder's x and y coordinates
            offset = self.n_particles * 4
            x_idx = offset + i * 2
            y_idx = offset + i * 2 + 1
            x = self.cached_state[x_idx]
            y = self.cached_state[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.boulder_radius / self.scale)
            pygame.draw.circle(self.screen, BOULDER_COLOR, center, radius)

        # Draw landmarks
        for i in range(self.n_landmarks):
            # Calculate the index for the landmark's x and y coordinates
            offset = self.n_particles * 4 + self.n_boulders * 2
            x_idx = offset + i * 2
            y_idx = offset + i * 2 + 1
            x = self.cached_state[x_idx]
            y = self.cached_state[y_idx]
            center = self.scale_to_screen(x, y)
            radius = int(self.landmark_radius / self.scale)
            pygame.draw.circle(self.screen, LANDMARK_COLOR, center, radius)

        # Update the display to show the changes
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(self.fps)

    def close(self):
        """Called to clean up resources."""
        print("Closing environment.")
        # If your C++ backend needs explicit cleanup (e.g., closing files,
        # freeing memory), you would call that here.
        pass


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    # --- VERIFY THE ENVIRONMENT WITH THE OFFICIAL PETTINGZOO TEST ---
    print("Running PettingZoo API Test...")
    env = CoopPushEnv()
    parallel_api_test(env, num_cycles=1000)
    print("API Test Passed!")

    # --- EXAMPLE USAGE ---
    print("\n--- Running Example Usage ---")
    env = CoopPushEnv(render_mode="human")
    observations, infos = env.reset()

    for step in range(256):
        # Get random actions for each agent
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        print(f"\nStep {step + 1}")
        print(f"Actions: {actions}")

        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        if not env.agents:
            print("All agents are done. Resetting.")
            observations, infos = env.reset()

    env.close()

    # env = cooppush_cpp.Environment()
    # env.init([0.0, 1.0], [1.0, 2.0], [2.0, 3.0])
    # print(env)
