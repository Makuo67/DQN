import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import legacy as optimizers
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import hospital_env

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
dqn.load_weights('dqn_weights2.h5f')

# Simulate the agent in the environment
dqn.test(env, nb_episodes=5, visualize=True)
