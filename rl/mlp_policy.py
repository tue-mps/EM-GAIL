import torch
import torch.nn as nn
from torch.distributions import Independent
from torch.distributions.normal import Normal

class RLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RLPolicy, self).__init__()


        self.pol = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
            )
        self.pol[-1].weight.data *= 0.1
        self.pol[-1].bias.data *= 0.0


        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def forward(self, x):

        mean = self.pol(x)
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)
        return normal_dist


class RLValue(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(RLValue, self).__init__()


        self.val = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
        self.val[-1].weight.data *= 0.1
        self.val[-1].bias.data *= 0.0

    def forward(self, x):
        out = self.val(x)
        return out

