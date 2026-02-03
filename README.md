# Computer-Science-311-AI-Assignment-3

# RL-Balancer: Deep Q-Network CartPole Agent

This repository contains a Reinforcement Learning (RL) implementation of a Deep Q-Network (DQN) agent designed to solve the classic **CartPole-v1** balancing task.

## 📋 Project Overview
The goal of this project is to train an agent to balance a pole on a moving cart. The agent receives a reward of +1 for every timestep the pole remains upright. The episode ends if the pole tilts more than 15 degrees or the cart moves more than 2.4 units from the center.

- **Environment:** [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- **Algorithm:** Deep Q-Network (DQN)
- **Framework:** Stable Baselines3 & PyTorch

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Optional] Virtual environment (venv or conda)

### Setup
1. Clone this repository or download the source files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt