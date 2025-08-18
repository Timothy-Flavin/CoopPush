import gymnasium
import numpy as np
from gymnasium.spaces import Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ActionType, AgentID, ObsType
import cooppush_cpp


# =============================================================================
# MOCK C++ BACKEND (Simulates your Pybind11 module)
# =============================================================================
class Pybind11Backend:
    """
    This class is a stand-in for your actual C++ environment compiled with
    Pybind11. It manages the underlying state and physics of the environment.

    In your real implementation, you would replace this class with:
    `from my_cpp_project import CppEnvironment`
    """

    def __init__(self, n_particles: int, continuous_actions: bool):
        print("Initializing Mock C++ Backend...")
        self.n_particles = n_particles
        self.continuous_actions = continuous_actions
        self.state = None

    def reset(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Resets the C++ environment to a starting state.

        Returns:
            - A NumPy array representing the full global state.
            - A dictionary of initial observations for each agent.
        """
        print("Backend: reset() called.")
        # Example state: [p1_x, p1_y, p2_x, p2_y, ...]
        self.state = np.random.rand(self.n_particles * 2).astype(np.float32)

        # Example observations: each agent sees its own position
        observations = {
            f"particle_{i}": self.state[i * 2 : (i + 1) * 2]
            for i in range(self.n_particles)
        }
        return self.state, observations

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, dict, dict, dict, dict]:
        """
        Steps the C++ environment forward.

        Args:
            actions: A NumPy array of actions for all agents, concatenated.

        Returns:
            - The new global state.
            - A dictionary of new observations.
            - A dictionary of rewards.
            - A dictionary of terminations.
            - A dictionary of truncations (not used in this simple example).
        """
        print(f"Backend: step() called with actions:\n{actions}")

        # --- Simulate Physics ---
        # Move particles based on actions. This is where your core C++ logic lives.
        # For this example, we'll just add the action vector to the position.
        if self.continuous_actions:
            action_reshaped = actions.reshape(self.n_particles, 2)
            self.state += action_reshaped.flatten() * 0.1  # Apply scaled actions
        else:  # Discrete actions
            for i, action in enumerate(actions):
                if action == 1:
                    self.state[i * 2] += 0.1  # Right
                elif action == 2:
                    self.state[i * 2] -= 0.1  # Left
                elif action == 3:
                    self.state[i * 2 + 1] += 0.1  # Up
                elif action == 4:
                    self.state[i * 2 + 1] -= 0.1  # Down

        # --- Generate Return Values ---
        observations = {
            f"particle_{i}": self.state[i * 2 : (i + 1) * 2]
            for i in range(self.n_particles)
        }

        # Example reward: negative distance to origin
        rewards = {
            f"particle_{i}": -np.linalg.norm(self.state[i * 2 : (i + 1) * 2])
            for i in range(self.n_particles)
        }

        # Example termination: if any particle goes out of bounds [0, 1]
        is_out_of_bounds = np.any(self.state < 0) or np.any(self.state > 1)
        terminations = {
            f"particle_{i}": is_out_of_bounds for i in range(self.n_particles)
        }

        truncations = {f"particle_{i}": False for i in range(self.n_particles)}

        return self.state, observations, rewards, terminations, truncations


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
        n_particles: int = 3,
        continuous_actions: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        env = cooppush_cpp.Environment()
        env.init([0.0, 1.0], [1.0, 2.0], [2.0, 3.0])
        print(env)
        self.n_particles = n_particles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode

        # --- Instantiate the C++ Backend ---
        # In a real project, this would be your imported Pybind11 module.
        self.backend = Pybind11Backend(n_particles, continuous_actions)

        # --- PettingZoo API Requirements ---
        self.agents = [f"particle_{i}" for i in range(n_particles)]
        self.possible_agents = self.agents[:]

        # Define observation and action spaces for each agent
        # Each agent observes its own (x, y) position
        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(2,), dtype=np.float32)
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
                agent: gymnasium.spaces.Discrete(5) for agent in self.possible_agents
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
        initial_state, initial_obs = self.backend.reset()

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
        new_state, obs, rewards, terminations, truncations = self.backend.step(
            action_array
        )

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
                f"Current State (Timestep {self.backend.state is not None and len(self.agents)})"
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
