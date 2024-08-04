from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np


class HospitalEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(HospitalEnv, self).__init__()
        # 4 possible actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, 5, 1), dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros((5, 5, 1), dtype=np.float32)
        self.state[0, 0, 0] = 1  # agent's starting position
        self.state[2, 2, 0] = -1  # obstacle: doctor
        self.state[3, 3, 0] = -1  # obstacle: nurse
        self.state[4, 4, 0] = 2  # goal: medicine cabinet
        return self.state

    def step(self, action):
        agent_position = np.argwhere(self.state == 1)[0]
        self.state[agent_position[0], agent_position[1], 0] = 0

        if action == 0:  # up
            agent_position[0] = max(agent_position[0] - 1, 0)
        elif action == 1:  # down
            agent_position[0] = min(agent_position[0] + 1, 4)
        elif action == 2:  # left
            agent_position[1] = max(agent_position[1] - 1, 0)
        elif action == 3:  # right
            agent_position[1] = min(agent_position[1] + 1, 4)

        reward = -1
        done = False

        if self.state[agent_position[0], agent_position[1], 0] == 2:  # reached goal
            reward = 10
            done = True
        elif self.state[agent_position[0], agent_position[1], 0] == -1:  # hit obstacle
            reward = -10
            done = True

        self.state[agent_position[0], agent_position[1], 0] = 1
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state[:, :, 0])


# Register the environment
register(id='Hospital-v0', entry_point='hospital_env:HospitalEnv')
