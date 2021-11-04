import torch
import torch.nn as nn
from torch.distributions import Independent
from torch.distributions.normal import Normal

class EMGPolicy(nn.Module):
    def __init__(self, intention_dim, state_dim, action_dim, hidden_dim=64):
        super(EMGPolicy, self).__init__()

        self.intention_dim = intention_dim
        self.state_dim = state_dim

        self.pol1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
            )
        self.pol1[-1].weight.data *= 0.1
        self.pol1[-1].bias.data *= 0.0
        self.pol2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
            )
        self.pol2[-1].weight.data *= 0.1
        self.pol2[-1].bias.data *= 0.0

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def forward(self, x):
        sq = False
        if len(x.size()) == 1:
            sq = True
            x = x.unsqueeze(0)
        intention = torch.narrow(x, -1, 0, self.intention_dim)
        state = torch.narrow(x, -1, self.intention_dim, self.state_dim)
        l = len(x.size())

        out = []
        out.append(self.pol1(state.clone()))
        out.append(self.pol2(state.clone()))
        out = torch.stack(out, dim=l-1)
        intention = intention.unsqueeze(l).repeat(1,1,out.size(-1))
        mean = torch.sum(out*intention, dim=l-1)
        if sq:
            mean = mean.squeeze(0)

        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)
        return normal_dist


class EMGValue(nn.Module):
    def __init__(self, intention_dim, state_dim, hidden_dim=64):
        super(EMGValue, self).__init__()

        self.intention_dim = intention_dim
        self.state_dim = state_dim

        self.val1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
        self.val1[-1].weight.data *= 0.1
        self.val1[-1].bias.data *= 0.0
        self.val2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
        self.val2[-1].weight.data *= 0.1
        self.val2[-1].bias.data *= 0.0

    def forward(self, x):
        intention = torch.narrow(x, -1, 0, self.intention_dim)
        state = torch.narrow(x, -1, self.intention_dim, self.state_dim+1)
        l = len(x.size())

        out = []
        out.append(self.val1(state.clone()))
        out.append(self.val2(state.clone()))
        out = torch.stack(out, dim=l-1)
        intention = intention.unsqueeze(l).repeat(1,1,out.size(-1))
        out = torch.sum(out*intention, dim=l-1)
        return out


class GPolicy(nn.Module):
    def __init__(self, intention_dim, state_dim, action_dim, hidden_dim=64):
        super(GPolicy, self).__init__()

        self.intention_dim = intention_dim
        self.state_dim = state_dim

        self.pol = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
            )
        self.pol[-1].weight.data *= 0.1
        self.pol[-1].bias.data *= 0.0

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def forward(self, x):
        x = torch.narrow(x, -1, self.intention_dim, self.state_dim)
        mean = self.pol(x)
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)
        return normal_dist


class GValue(nn.Module):
    def __init__(self, intention_dim, state_dim, hidden_dim=64):

        super(GValue, self).__init__()

        self.intention_dim = intention_dim
        self.state_dim = state_dim

        self.val = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
            )
        self.val[-1].weight.data *= 0.1
        self.val[-1].bias.data *= 0.0


    def forward(self, x):
        x = torch.narrow(x, -1, self.intention_dim, self.state_dim+1)

        out = self.val(x)
        return out
