# 1) Import the necessary packages. For this tutorial, we will use the `quadruped_locomotion` environment. 
from a2perf.domains.web_navigation.gwob.CoDE import environment
# import the relevant A2Perf domain
import a2perf
# import a2perf.domains.quadruped_locomotion
import a2perf.domains.web_navigation.gwob.CoDE
# import gymnasium to create the environment
import gymnasium as gym
from gymnasium import spaces
from typing import Callable
import gin

import snntorch as snn
from snntorch import surrogate
# import the abseil app to run the experiment
from absl import app

# import packages needed for your training
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO

import torch
import torch.nn as nn

import time
# print all registered gym environments
print('Creating environment')
env = gym.make('WebNavigation-v0',num_websites=2, difficulty=2, raw_state=True,step_limit=7)
# env = gym.make('QuadrupedLocomotion-v0')


env_low = env.env.unwrapped
miniwobstate = env.reset()[0]
print(miniwobstate)
print('Utterance')
print(miniwobstate.utterance)
print('Phrase')
print(miniwobstate.phrase)
print('Tokens')
print(miniwobstate.tokens)
print('Fields')
print(miniwobstate.fields)
print('Dom')
print(miniwobstate.dom)
print('Dom elements')
print(miniwobstate.dom_elements)
# env.reset(options={'initial_motor_angles':None})
# env.reset(options={'raw_state': False})

time.sleep(1)
print(env.step(0))
# print(out[0]['dom_profile_joint_mask'].shape)
# print(out[0]['dom_elements'].shape)

# print(out[0]['profile_value'].shape)
# env_spec = gym.spec('QuadrupedLocomotion-v0')
# print(env_spec.kwargs)
# print(env.action_space)
# print(env.observation_space)

class CustomNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(CustomNet, self).__init__()

        self.latent_dim_pi = action_shape.n
        self.latent_dim_vf = 1
        self.actor = nn.Sequential(
            nn.Linear(state_shape.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape.n),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_shape.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
          return self.actor(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features)
    
    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

class CustomSNN(nn.Module):
  def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
      super(CustomSNN, self).__init__()
      self.spike_grad = surrogate.FastSigmoid.apply
      num_outputs = action_space.n

      beta = 0.95
      self.nr_passes = nr_passes

      self.lin1 = nn.Linear(num_inputs, 128)
      self.lif1 = snn.Leaky(beta = .95, spike_grad=self.spike_grad, learn_beta=True)
      self.lin2 = nn.Linear(128, 128)
      self.lif2 = snn.Leaky(beta = beta, spike_grad=self.spike_grad, learn_beta=True)

      num_outputs = action_space.n
      self.critic_linear = nn.Linear(128, 1)
      self.actor_linear = nn.Linear(128, num_outputs)
    
      # membranes at t = 0
      self.mem1     = self.lif1.init_leaky()
      self.mem2     = self.lif2.init_leaky()

      self.inp_min = inp_min
      self.inp_max = inp_max

      self.train()

      self.spk_in_rec = []  # Record the output trace of spikes
      self.mem_in_rec = []  # Record the output trace of membrane potential

      self.spk1_rec = []  # Record the output trace of spikes
      self.mem1_rec = []  # Record the output trace of membrane potential
      
      self.spk2_rec = []  # Record the output trace of spikes
      self.mem2_rec = []  # Record the output trace of membrane potential


      self.spk3_rec = []  # Record the output trace of spikes
      self.mem3_rec = []  # Record the output trace of membrane potential
      
  def init_mem(self):
      self.mem1     = self.lif1.init_leaky()
      self.mem2     = self.lif2.init_leaky()

      self.spk_in_rec = []  # Record the output trace of spikes
      self.mem_in_rec = []  # Record the output trace of membrane potential

      self.spk1_rec = []  # Record the output trace of spikes
      self.mem1_rec = []  # Record the output trace of membrane potential
      
      self.spk2_rec = []  # Record the output trace of spikes
      self.mem2_rec = []  # Record the output trace of membrane potential

      self.spk3_rec = []  # Record the output trace of spikes
      self.mem3_rec = []  # Record the output trace of membrane potential


  def forward(self, inputs, nr_passes = 1):
      
      for i in range(self.nr_passes):
          inputs = torch.tensor(inputs).to(torch.float32)
          inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

          inputs = inputs.to(torch.float32)

          # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
          cur1 = self.lin1(inputs)
          spk1, self.mem1 = self.lif1(cur1, self.mem1)

          cur2 = self.lin2(spk1)
          spk2, self.mem2 = self.lif2(cur2, self.mem2)



      actions =  self.actor_linear(spk1)
      
      val = self.critic_linear(spk1)
      
      # add information for plotting purposes
      self.spk_in_rec.append(spk1)  # Record the output trace of spikes
      self.mem_in_rec.append(self.mem1)  # Record the output trace of membrane potential

      self.spk1_rec.append(spk2)  # Record the output trace of spikes
      self.mem1_rec.append(self.mem2)  # Record the output trace of membrane potential


      return val, actions
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    def _build_mlp_extractor(self) -> None:
      self.mlp_extractor = CustomNet(self.observation_space, self.action_space)


# 2) Next, we define our training function. This function will be called by the abseil app.
def train():
    # ac = CustomActorCriticPolicy(env.observation_space.shape, env.action_space)
    '''Include your training algorithm here.'''
    # Create the environment
    vec_env = make_vec_env("CartPole-v1", n_envs=8)

    # Create the agent
    model = PPO(CustomActorCriticPolicy, vec_env, verbose=1)
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
#   app.run(main)
  pass