import pygame
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import legacy as optimizers
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import hospital_env

# Initialize Pygame
pygame.init()
screen_size = (500, 500)
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

env = gym.make('Hospital-v0')
actions = env.action_space.n
states = env.observation_space.shape


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
model.summary()


def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

# Load the trained weights
dqn.load_weights('dqn_weights.h5f')

# Visualization function


def render(env):
    state = env.reset()
    done = False
    while not done:
        screen.fill((255, 255, 255))  # White background
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action = dqn.forward(env.state)
        next_state, reward, done, _ = env.step(action)
        # Render the environment
        screen.fill((255, 255, 255))  # Background
        for i in range(5):
            for j in range(5):
                value = env.state[i, j, 0]
                color = (0, 0, 255) if value == 1 else (
                    255, 0, 0) if value == -1 else (0, 255, 0) if value == 2 else (255, 255, 255)
                pygame.draw.rect(screen, color, pygame.Rect(
                    j * 100, i * 100, 100, 100))

        pygame.display.flip()
        clock.tick(1)  # 1 FPS

        env.state = next_state


# Run the visualization
render(env)
