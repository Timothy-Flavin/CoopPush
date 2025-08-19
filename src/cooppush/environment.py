import gymnasium
import numpy as np
import json
from gymnasium.spaces import Box

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
    ):
        super().__init__()
        self.continuous_actions = True
        with open(json_path) as f:
            env_setup = json.load(f)

        env = cooppush_cpp.Environment()
        env.init(
            env_setup["particle_pos"],
            env_setup["boulder_pos"],
            env_setup["landmark_pos"],
        )
        print(env)
        self.env = env
        self.n_particles = len(env_setup["particle_pos"]) // 2
        self.n_boulders = len(env_setup["boulder_pos"]) // 2
        self.n_landmarks = len(env_setup["landmark_pos"]) // 2

        self.render_mode = render_mode

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
                agent: Box(
                    low=-1, high=1, shape=(self.n_particles * 2,), dtype=np.float32
                )
                for agent in self.possible_agents
            }
        else:
            # 0: no-op, 1: right, 2: left, 3: up, 4: down
            self.action_spaces = {
                agent: gymnasium.spaces.Discrete(self.n_particles * 2)
                for agent in self.possible_agents
            }

        # --- State Caching for Rendering ---
        # This variable will hold the full state returned by the C++ backend
        # so the `render` function can use it without making another C++ call.
        self.cached_state = None

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
        # --- 1. Format actions for the backend ---
        # The backend expects a single, ordered NumPy array.
        # We must ensure actions are in the correct agent order.
        ordered_actions = []
        for agent in self.possible_agents:
            if agent in self.agents:
                ordered_actions.append(actions[agent])
            else:  # If an agent is done, provide a default action (e.g., no-op)
                if self.continuous_actions:
                    ordered_actions.append(np.zeros(2, dtype=np.float32))
                else:
                    ordered_actions.append(0)

        action_array = np.array(ordered_actions)

        # --- 2. Call the backend ---
        new_state, obs, rewards, terminations, truncations = self.env.step(action_array)

        # --- 3. Cache the new state ---
        self.cached_state = new_state

        # --- 4. Format results for PettingZoo ---
        # Handle agent termination
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                # This agent is now done
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

    def render(self) -> None | str:
        """
        Renders the environment using the cached state.
        This method DOES NOT call the C++ backend.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.cached_state is None:
            print("Cannot render, state is not initialized. Call reset() first.")
            return

        if self.render_mode in ["human", "ansi"]:
            # Simple text-based rendering
            print("-" * 20)
            print(
                f"Current State (Timestep {self.env.state is not None and len(self.agents)})"
            )
            for i in range(self.n_particles):
                x = self.cached_state[i * 2]
                y = self.cached_state[i * 2 + 1]
                print(f"  Particle {i}: (x={x:.3f}, y={y:.3f})")
            print("-" * 20)
            if self.render_mode == "ansi":
                return "Rendering output as a string would go here."

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
    env = CoopPushEnv(n_particles=3, continuous_actions=True)
    parallel_api_test(env, num_cycles=1000)
    print("API Test Passed!")

    # --- EXAMPLE USAGE ---
    print("\n--- Running Example Usage ---")
    env = CoopPushEnv(n_particles=2, continuous_actions=True, render_mode="human")
    observations, infos = env.reset()

    for step in range(5):
        # Get random actions for each agent
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        print(f"\nStep {step + 1}")
        print(f"Actions: {actions}")

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if not env.agents:
            print("All agents are done. Resetting.")
            observations, infos = env.reset()

    env.close()

    env = cooppush_cpp.Environment()
    env.init([0.0, 1.0], [1.0, 2.0], [2.0, 3.0])
    print(env)
