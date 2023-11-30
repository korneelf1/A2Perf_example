import rl_perf
import rl_perf.domains.quadruped_locomotion

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from absl import app



def train():
    '''Include your training algorithm here.'''
    # Create the environment
    vec_env = make_vec_env("QuadrupedLocomotion-v0", n_envs=8)

    # Create the agent
    model = PPO("MlpPolicy", vec_env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=25e3)
    # Save the agent
    model.save("ppo_cartpole")

    del model # remove to demonstrate saving and loading

def main(_):
  # The main function where the training process is initiated.
  train()


if __name__ == '__main__':
  app.run(main)