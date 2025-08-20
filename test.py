from cooppush.environment import CoopPushEnv
import pygame
import numpy as np
from pettingzoo.test import parallel_api_test


# print("Running PettingZoo API Test...")
# env = CoopPushEnv()
# parallel_api_test(env, num_cycles=1000)
# print("API Test Passed!")


keys = {"w": False, "a": False, "s": False, "d": False}


def handle_input(keys):
    dx, dy = 0, 0
    if keys["w"]:
        dy = -1
    if keys["s"]:
        dy = 1
    if keys["a"]:
        dx = -1
    if keys["d"]:
        dx = 1
    return [dx, dy]


def handle_event(keys):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for key presses
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                keys["w"] = True
            if event.key == pygame.K_a:
                keys["a"] = True
            if event.key == pygame.K_s:
                keys["s"] = True
            if event.key == pygame.K_d:
                keys["d"] = True

        # Check for key releases
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                keys["w"] = False
            if event.key == pygame.K_a:
                keys["a"] = False
            if event.key == pygame.K_s:
                keys["s"] = False
            if event.key == pygame.K_d:
                keys["d"] = False


env = CoopPushEnv(
    render_mode="human",
    fps=20,
    sparse_rewards=False,
    visit_all=True,
    randomize_order=True,
    start_noise=0.5,
)
observations, infos = env.reset()

terminated = False
while not terminated:
    # Get random actions for each agent
    # print(env.agents)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # print(f"\nStep {step + 1}")
    # print(f"Actions: {actions}")
    handle_event(keys)
    hact = handle_input(keys)
    actions["particle_0"] = np.array(hact)

    observations, rewards, terminations, truncations, infos = env.step(actions)
    terminated = terminations["particle_0"]
    env.render()
    # print(terminations)
    if not env.agents:
        print("All agents are done. Resetting.")
        observations, infos = env.reset()

print("Resseting as a test")
env.reset()
print("Resseting again as a test")
env.reset()
env.close()
