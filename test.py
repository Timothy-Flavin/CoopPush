from cooppush.environment import CoopPushEnv
import pygame
import numpy as np
from pettingzoo.test import parallel_api_test


# print("Running PettingZoo API Test...")
# env = CoopPushEnv()
# parallel_api_test(env, num_cycles=1000)
# print("API Test Passed!")


keys = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "i": False,
    "j": False,
    "k": False,
    "l": False,
}


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


def handle_input2(keys):
    dx, dy = 0, 0
    if keys["i"]:
        dy = -1
    if keys["k"]:
        dy = 1
    if keys["j"]:
        dx = -1
    if keys["l"]:
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
            if event.key == pygame.K_i:
                keys["i"] = True
            if event.key == pygame.K_j:
                keys["j"] = True
            if event.key == pygame.K_k:
                keys["k"] = True
            if event.key == pygame.K_l:
                keys["l"] = True
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
            if event.key == pygame.K_i:
                keys["i"] = False
            if event.key == pygame.K_j:
                keys["j"] = False
            if event.key == pygame.K_k:
                keys["k"] = False
            if event.key == pygame.K_l:
                keys["l"] = False
    pygame.event.clear()


env = CoopPushEnv(
    render_mode="human",
    cpp_steps_per_step=10,
    fps=3,
    sparse_rewards=False,
    visit_all=True,
    randomize_order=True,
    start_noise=0.5,
    normalize_observations=True,
    dt=0.2,
    boulder_weight=4.0,
)
observations, infos = env.reset()

#'physics_steps': 10, 'sparse_rewards': False, 'randomize_order': True, 'start_noise': 1.0, 'level_name': './levels/dependent.json', 'visit_all': True
n_agents = len(env.agents)
print(observations["particle_0"].shape)
print(env.observation_space("particle_0").shape)
# input("this make sense?")
terminated = False
step = 1
while not terminated:
    # Get random actions for each agent
    # print(env.agents)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # print(f"\nStep {step + 1}")
    # print(f"Actions: {actions}")
    print(observations["particle_0"][0:4])
    handle_event(keys)
    hact = handle_input(keys)
    hact2 = handle_input2(keys)
    actions["particle_0"] = np.array(hact)
    actions["particle_1"] = np.array(hact2)

    observations, rewards, terminations, truncations, infos = env.step(actions)
    terminated = terminations["particle_0"]
    env.render(importance=np.arange(n_agents) / n_agents)
    # print(terminations)
    step += 1
    if not env.agents:
        print("All agents are done. Resetting.")
        observations, infos = env.reset()

print(f"Num Steps to complete: {step}")
print("Resseting as a test")
env.reset()
print("Resseting again as a test")
env.reset()
env.close()
