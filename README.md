# Doctor's Assistant Training Project

## Overview

This project involves training a Doctor's Assistant in a medical emergency scenario where the assistant delivers necessary surgical equipment, medicines, etc., to the doctor. The assistant must avoid obstacles such as doctors, nurses, and hospital beds while navigating back and forth in the environment.

## Project Structure

The project consists of the following files:

1. `hospital_env.py`: Defines the custom gym environment for the hospital scenario.
2. `train.py`: Trains a Deep Q-Network (DQN) agent to navigate the hospital environment.
3. `play.py`: Loads the trained model and simulates the agent in the environment.
4. `visualize.py`: Uses Pygame to visualize the agent's actions in the environment.

## Files Description

### 1. hospital_env.py

Defines the custom Gym environment for the hospital scenario, including the state space, action space, and reward mechanism.

### 2. train.py

Trains a Deep Q-Network (DQN) agent using the custom hospital environment. The agent learns to navigate the environment by interacting with it and receiving rewards for reaching the goal and penalties for hitting obstacles.

### 3. play.py

Loads the trained model and runs it in the hospital environment to demonstrate the agent's performance. It simulates the agent's behavior using the learned policy.

### 4. visualize.py

Visualizes the agent's actions in the hospital environment using Pygame. It provides a graphical representation of the environment and the agent's movement.

## Running the Project

### Prerequisites

- Python 3.6+
- Gym
- Numpy
- Keras
- TensorFlow
- Pygame
- Keras-RL2

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Makuo67/DQN.git
   cd DQN
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Agent

To train the agent, run:

```bash
python train.py
```

To simulate the trained agent, run:

```bash
python play.py
```

To visualize the agent's actions using Pygame, run:

```bash
python visualize.py
```

## Demonstration

Watch the demo of the project on YouTube: [Demo Video](https://www.youtube.com/watch?v=r5rRPuKl84E)

<iframe width="560" height="315" src="https://www.youtube.com/embed/r5rRPuKl84E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
