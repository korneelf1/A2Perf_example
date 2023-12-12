import tianshou as ts
import gymnasium as gym
import torch
from actor_critics import Actor_net, Critic_net
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env.venvs import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer


env = gym.make("QuadrupedLocomotion-v0")
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor_net(state_shape, action_shape)
critic = Critic_net(state_shape, action_shape)

actor = ActorProb(
        Actor_net,
        env.action_space.shape,
        unbounded=True,
        device=device,
    ).to(device)

critic = Critic(Critic_net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)

optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
policy = ts.policy.PPOPolicy(actor, critic, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
train_epoch = 100

# Initialize collector and buffer
buffer_size = 10000
train_collector = Collector(policy, env, buffer_size)
train_collector.collect(n_step=buffer_size)


logger = TensorboardLogger()
# Training loop
result = onpolicy_trainer(
    policy, train_collector, max_epoch=train_epoch, episode_per_epoch=10,
    step_per_epoch=buffer_size, repeat_per_collect=5,
    buffer_size=buffer_size, logger=logger, test_env=env
)