import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os

# In this section we create the environment and utilize CartPole-v1 with the goal to balance a pole by moving the cart left/right
env = gym.make("CartPole-v1", render_mode="human")
def agent_training():
    print("Training process has been initiated..")
    # In this section we define the model
    model = DQN("MlpPolicy", env, verbose=1, 
                learning_rate=1e-3, 
                buffer_size=10000, 
                learning_starts=1000, 
                target_update_interval=500)

    # In this section we train the agent on the total steps
    model.learn(total_timesteps=10000)
    model.save("dqn_cartpole_model")
    print("Model Saved.")
    return model

def testing_agent(model):
    print("Test for agent has been initiated..")
    obs, info = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    # Check if a model exists, otherwise train a new one
    if not os.path.exists("dqn_cartpole_model.zip"):
        trained_model = agent_training()
    else:
        trained_model = DQN.load("dqn_cartpole_model", env=env)
    
    testing_agent(trained_model)