import torch

class Actor_net(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor_net, self).__init__()
        self.lin_layers = torch.nn.Sequential(
            torch.nn.Linear(state_shape, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
        )
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(288, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_shape),
            torch.nn.Tanh(),
        )

    def forward(self, obs, info={}):
        out_lin = self.lin_layers(obs.float())
        out_conv = self.conv_layers(out_lin)
        return out_conv
    
class Critic_net(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Critic_net, self).__init__()
        self.lin_layers = torch.nn.Sequential(
            torch.nn.Linear(state_shape, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
        )
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(288, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_shape),
            torch.nn.Tanh(),
        )

    def forward(self, obs, info={}):
        out_lin = self.lin_layers(obs.float())
        out_conv = self.conv_layers(out_lin)
        return out_conv