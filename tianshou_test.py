import gymnasium as gym
import torch
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

import actor_critics as ac

import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch import nn



class SNN_LIF(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden1=32,hidden2=32, inp_min = torch.tensor([0]), inp_max=  torch.tensor([2.5]),bias=False, nr_passes = 1 ):
        super(SNN_LIF, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        self.num_outputs = action_space.n
        self.num_inputs = num_inputs
        beta = 0.95
        self.nr_passes = nr_passes

        # randomly initialize decay rate and threshold for layer 1
        beta_hidden = torch.rand(hidden1)
        thr_hidden = torch.rand(hidden1)

        self.lin1 = nn.Linear(num_inputs, hidden1)
        self.lif1 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(hidden2)
        thr_hidden = torch.rand(hidden2)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)
        # self.lif2 = snn.Synaptic(beta = .75, alpha = 0.5, spike_grad=self.spike_grad, learn_beta=False, learn_alpha=False)

        self.lin_out = nn.Linear(hidden2, self.num_outputs)
        self.lif_out = snn.Leaky(beta = torch.rand(self.num_outputs), spike_grad=self.spike_grad, learn_beta=True)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.

       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.memout     = self.lif_out.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []

    def init_mem(self):
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.memout     = self.lif_out.init_leaky()
        
        self.inputs = []

    def clip_hiddens(self):
        self.lif1.beta = torch.nn.Parameter(torch.clamp(self.lif1.beta, min=0, max=1))
        self.lif2.beta = torch.nn.Parameter(torch.clamp(self.lif2.beta, min=0, max=1))
        self.lif_out.beta = torch.nn.Parameter(torch.clamp(self.action_lif.beta, min=0, max=1))

    def forward(self, inputs, state=None, nr_passes = 1):
        if state is None:
            self.init_mem()
        else:
            self.mem1 = state["hidden"]["layer1"]
            self.mem2 = state["hidden"]["layer2"]
            self.memout = state["hidden"]["layerout"]

        for i in range(self.nr_passes):
            inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)
            self.inputs.append(inputs)
            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            cur_out = self.lin_out(spk2)
            _, self.memout = self.lif_out(cur_out, self.memout)
        hiddens = {"layer1":self.mem1,
                   "layer2": self.mem2, 
                   "layerout":self.memout}
        return self.memout, hiddens


# Define the neural network model
class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape. """

    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        hidden_size1: int = 64,
        hidden_size2: int = 64,
        softmax_output: bool = True,

        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = int(np.prod(action_shape.n))
        
        self.softmax_output = softmax_output

        self.net = SNN_LIF(input_dim, action_shape, hidden1=hidden_size1, hidden2=hidden_size2, nr_passes=1)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        # logits, hidden = self.preprocess(obs, state)
        logits, hidden = self.net(obs, state=state)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)

        return logits, {"hidden":hidden}


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        hidden_size1: int = 64,
        hidden_size2: int = 64,
        softmax_output: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.net = SNN_LIF(input_dim, action_shape, hidden1=hidden_size1, hidden2=hidden_size2, nr_passes=1)

    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
                # logits, hidden = self.preprocess(obs, state)
        logits, hidden = self.net(obs, state=kwargs.get("state", None))

        # return logits, {"hidden":hidden}
        return logits


# # environments
env = gym.make("CartPole-v1")
train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

# model & optimizer
# net = Net(env.observation_space.shape, hidden_sizes=[32, 32], device=device)
# actor = Actor(net, env.action_space.n, device=device).to(device)
# critic = Critic(net, device=device).to(device)
actor = Actor(env.observation_space.shape[0], env.action_space, hidden_size1=32, hidden_size2=32, softmax_output=True, device=device).to(device)
critic = Critic(env.observation_space.shape[0], env.action_space, hidden_size1=32, hidden_size2=32, softmax_output=False, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0007)

# PPO policy
dist = torch.distributions.Categorical
policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    action_space=env.action_space,
    action_scaling=False,
)


# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs),stack_num=100))
test_collector = Collector(policy, test_envs)

# trainer
result = OnpolicyTrainer(
    policy=policy,
    batch_size=256,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=10,
    step_per_epoch=50000,
    repeat_per_collect=10,
    episode_per_test=10,
    step_per_collect=2000,
    stop_fn=lambda mean_reward: mean_reward >= 198,
).run()
print(result)


# Let's watch its performance!
policy.eval()
result = test_collector.collect(n_episode=1, render=False)
print("Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))
print(result)