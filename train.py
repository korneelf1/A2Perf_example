# 1) Import the necessary packages. For this tutorial, we will use the `quadruped_locomotion` environment. 

# import the relevant A2Perf domain
import rl_perf
import rl_perf.domains.quadruped_locomotion
import rl_perf.domains.web_nav
# import gymnasium to create the environment
import gymnasium as gym

# import the abseil app to run the experiment
from absl import app

# import packages needed for your training
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# print all registered gym environments
env = gym.make('WebNavigation-v0', {'headless':True})
env.reset()
# 2) Next, we define our training function. This function will be called by the abseil app.
def train():
    '''Include your training algorithm here.'''
    # Create the environment
    vec_env = make_vec_env("WebNavigation-v0", n_envs=8)

    # Create the agent
    model = PPO("MlpPolicy", vec_env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=25e3)
    # Save the agent
    model.save("ppo_cartpole")

    del model # remove to demonstrate saving and loading

# 3) Optionally, we define the main function. This function will be called when the script is run directly.
def main(_):
  # The main function where the training process is initiated.
  train()


if __name__ == '__main__':
  # Run the main function using the abseil app. This allows us to pass command line arguments to the script.
  app.run(main)